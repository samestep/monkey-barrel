mod barrel;
mod lbfgs;
mod monkey;
mod util;

use minkowski::{extract_loops, reduced_convolution, Point};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use resvg::{
    render,
    tiny_skia::Pixmap,
    usvg::{
        fontdb::Database, Options, PostProcessingSteps, Transform, Tree, TreeParsing, TreePostProc,
    },
};
use std::{
    fmt,
    fs::{create_dir_all, File},
    io::Write as _,
    path::Path,
};
use util::{cross, dot, vec2, Vec2};

const BARREL_BEZIER: &str = include_str!("BarrelCubicBézier.txt");
const MONKEY_BEZIER: &str = include_str!("MonkeyCubicBézier.txt");

const WIDTH: f64 = 1554.197998;
const HEIGHT: f64 = 2507.9672852;
const GAP: f64 = 10.;

const EPS: f64 = 1e-3;

type Polygon = Vec<Vec2>;

fn signed_area(p: &[Vec2]) -> f64 {
    let n = p.len();
    let mut sum = 0.;
    let mut u = p[n - 1];
    for &v in p {
        sum += cross(u, v);
        u = v;
    }
    sum / 2.
}

#[derive(Debug, Eq, PartialEq)]
enum Orientation {
    Clockwise,
    Counterclockwise,
}

fn orientation(p: &[Vec2]) -> Orientation {
    if signed_area(p) < 0. {
        Orientation::Clockwise
    } else {
        Orientation::Counterclockwise
    }
}

fn sd_polygon(q: &[Vec2], p: Vec2) -> (f64, Vec2) {
    let n = q.len();
    let mut dd = f64::INFINITY;
    let mut e = true;
    let mut i = n - 1;
    let mut k = (i, 0);
    let mut s = false;
    let mut v0 = q[i] - q[i - 1];
    for j in 0..n {
        let u = p - q[i];
        let v = q[j] - q[i];
        let vv = dot(v, v);
        let t = dot(u, v);
        if 0. <= t && t < vv {
            let w = cross(u, v);
            let dd1 = w * w / vv;
            if dd1 < dd {
                dd = dd1;
                e = true;
                k = (i, j);
            }
        } else {
            let dd1 = dot(u, u);
            if dd1 < dd {
                dd = dd1;
                e = false;
                k = (i, j);
                s = cross(v0, v) < 0.;
            }
        }
        i = j;
        v0 = v;
    }
    let (i, j) = k;
    let u = p - q[i];
    if e {
        let v = q[j] - q[i];
        let z = v.norm();
        (cross(u, v) / z, vec2(v.y, -v.x) / z)
    } else {
        let d = dd.sqrt();
        let g = u / d;
        if s {
            (-d, -g)
        } else {
            (d, g)
        }
    }
}

struct Monkeys {
    coords: Vec<f64>,
}

fn init(rng: &mut impl Rng, n: usize) -> Monkeys {
    let mut coords: Vec<_> = (0..n).map(|_| rng.gen_range(0.0..WIDTH)).collect();
    coords.extend((0..n).map(|_| rng.gen_range(0.0..HEIGHT)));
    Monkeys { coords }
}

fn val_and_grad(rng: &mut impl Rng, coords: &[f64], grad: &mut [f64]) -> f64 {
    grad.fill(0.);
    let n = coords.len() / 2;
    let (xs, ys) = coords.split_at(n);
    let (dxs, dys) = grad.split_at_mut(n);
    let mut fx = 0.;

    let barr: Vec<Point> = barrel::POLYGON
        .into_iter()
        .map(|Vec2 { x, y }| (x + rng.gen_range(-EPS..EPS), y + rng.gen_range(-EPS..EPS)))
        .collect();
    for i in 0..n {
        let monk: Vec<Point> = monkey::POLYGON
            .iter()
            .map(|Vec2 { x, y }| {
                (
                    -(xs[i] + x + rng.gen_range(-EPS..EPS)),
                    -(ys[i] + y + rng.gen_range(-EPS..EPS)),
                )
            })
            .collect();
        let mut loops: Vec<_> = extract_loops(&reduced_convolution(&barr, &monk))
            .into_iter()
            .map(|p| p.into_iter().map(|((x, y), _)| Vec2 { x, y }).collect())
            .filter(|p: &Polygon| orientation(p) == Orientation::Clockwise)
            .collect();
        assert_eq!(loops.len(), 1);
        let sum: Polygon = loops.swap_remove(0);
        let (z, dp) = sd_polygon(&sum, vec2(0., 0.));
        let w = GAP - z;
        if w > 0. {
            fx += w * w;
            dxs[i] -= 2. * w * dp.x;
            dys[i] -= 2. * w * dp.y;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let a: Vec<Point> = monkey::POLYGON
                .iter()
                .map(|Vec2 { x, y }| {
                    (
                        xs[i] + x + rng.gen_range(-EPS..EPS),
                        ys[i] + y + rng.gen_range(-EPS..EPS),
                    )
                })
                .collect();
            let b: Vec<Point> = monkey::POLYGON
                .iter()
                .map(|Vec2 { x, y }| {
                    (
                        -(xs[j] + x + rng.gen_range(-EPS..EPS)),
                        -(ys[j] + y + rng.gen_range(-EPS..EPS)),
                    )
                })
                .collect();
            let mut loops: Vec<_> = extract_loops(&reduced_convolution(&a, &b))
                .into_iter()
                .map(|p| p.into_iter().map(|((x, y), _)| Vec2 { x, y }).collect())
                .filter(|p: &Polygon| orientation(p) == Orientation::Counterclockwise)
                .collect();
            assert_eq!(loops.len(), 1);
            let c = loops.swap_remove(0);
            let (z, dp) = sd_polygon(&c, vec2(0., 0.));
            let w = GAP - z;
            if w > 0. {
                fx += w * w;
                dxs[i] += 2. * w * dp.x;
                dys[i] += 2. * w * dp.y;
                dxs[j] -= 2. * w * dp.x;
                dys[j] -= 2. * w * dp.y;
            }
        }
    }

    fx
}

fn optimize(
    rng: &mut impl Rng,
    mut monkeys: Monkeys,
    mut callback: impl FnMut(Option<&lbfgs::Info>, &[f64]),
) -> (Monkeys, f64) {
    callback(None, &monkeys.coords);
    let cfg = lbfgs::Config {
        m: 17,
        armijo: 0.001,
        wolfe: 0.9,
        min_interval: 1e-9,
        max_steps: 10,
        epsd: 1e-11,
    };
    let mut state = lbfgs::first_step(
        cfg,
        |coords, grad| val_and_grad(rng, coords, grad),
        &mut monkeys.coords,
    );
    callback(None, &monkeys.coords);
    let mut fx = f64::NAN;
    lbfgs::step_until(
        cfg,
        |coords, grad| val_and_grad(rng, coords, grad),
        &mut monkeys.coords,
        &mut state,
        |info| {
            callback(Some(&info), info.x);
            if info.fx == fx {
                Some(())
            } else {
                fx = info.fx;
                None
            }
        },
    );
    (monkeys, fx)
}

fn arrangement(w: &mut impl fmt::Write, coords: &[f64]) -> fmt::Result {
    let n = coords.len() / 2;
    writeln!(
        w,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}">"#,
    )?;
    writeln!(w, "  <defs>")?;
    writeln!(
        w,
        r##"    <path id="monkey" fill="#FF0000" stroke="#770000" stroke-width="4" d="{}" />"##,
        MONKEY_BEZIER,
    )?;
    writeln!(w, "  </defs>")?;
    writeln!(
        w,
        r##"  <path fill="#FFF200" stroke="#FFAA00" stroke-width="10" d="{}" />"##,
        BARREL_BEZIER,
    )?;
    for i in 0..n {
        writeln!(
            w,
            r##"  <use x="{}" y="{}" href="#monkey" />"##,
            coords[i],
            coords[n + i],
        )?;
    }
    writeln!(w, "</svg>")?;
    Ok(())
}

fn rasterize(svg: &str) -> Pixmap {
    let mut tree = Tree::from_str(svg, &Options::default()).unwrap();
    tree.postprocess(PostProcessingSteps::default(), &Database::new());
    let mut pixmap = Pixmap::new(tree.size.width() as u32, tree.size.height() as u32).unwrap();
    render(&tree, Transform::identity(), &mut pixmap.as_mut());
    pixmap
}

fn run(dir: &Path, n: usize, seed: u64) -> f64 {
    let dir_frames = dir.join(format!("{n}-{seed}"));
    create_dir_all(&dir_frames).unwrap();
    let mut i: usize = 0;
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let monkeys = init(&mut rng, n);
    let (Monkeys { coords }, fx) = optimize(&mut rng, monkeys, |info, coords| {
        if i.count_ones() < 2 {
            print!("i = {i}");
            if let Some(info) = info {
                println!(", fx = {}", info.fx);
            } else {
                println!();
            }
            let mut s = String::new();
            arrangement(&mut s, coords).unwrap();
            File::create(dir_frames.join(format!("{i}.svg")))
                .unwrap()
                .write_all(s.as_bytes())
                .unwrap();
            rasterize(&s)
                .save_png(dir_frames.join(format!("{i}.png")))
                .unwrap();
        }
        i += 1;
    });
    i -= 1;
    println!("i = {i}, fx = {fx}");
    let mut s = String::new();
    arrangement(&mut s, &coords).unwrap();
    File::create(dir_frames.join(format!("{i}.svg")))
        .unwrap()
        .write_all(s.as_bytes())
        .unwrap();
    rasterize(&s)
        .save_png(dir_frames.join(format!("{i}.png")))
        .unwrap();
    fx
}

fn main() {
    assert_eq!(orientation(&barrel::POLYGON), Orientation::Clockwise);
    assert_eq!(orientation(&monkey::POLYGON), Orientation::Counterclockwise);
    let dir = Path::new("out");
    run(dir, 10, 2);
}
