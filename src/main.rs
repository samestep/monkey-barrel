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
    f64::consts::TAU,
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

fn pointize(p: &[Vec2]) -> Vec<Point> {
    p.iter().map(|&Vec2 { x, y }| (x, y)).collect()
}

fn minkowski_sum(a: &[Vec2], b: &[Vec2], o: Orientation) -> Polygon {
    let edges = reduced_convolution(&pointize(a), &pointize(b));
    let mut loops = extract_loops(&edges);
    loops.sort_by_key(|p| p.len()); // TODO: actually compute nesting instead of this hack
    let c = loops.pop().unwrap();
    let poly: Polygon = c.iter().map(|&((x, y), _)| vec2(x, y)).collect();
    assert_eq!(orientation(&poly), o);
    poly
}

struct Monkeys {
    thetas: Vec<f64>,
    coords: Vec<f64>,
}

fn init(seed: u64, n: usize) -> Monkeys {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut coords: Vec<_> = (0..n).map(|_| rng.gen_range(0.0..WIDTH)).collect();
    coords.extend((0..n).map(|_| rng.gen_range(0.0..HEIGHT)));
    Monkeys {
        thetas: (0..n).map(|_| rng.gen_range(0.0..TAU)).collect(),
        coords,
    }
}

struct Sums {
    contains: Vec<Polygon>,
    pairs: Vec<Vec<Option<Polygon>>>,
}

fn val_and_grad(sums: &Sums, thetas: &[f64], coords: &[f64], grad: &mut [f64]) -> f64 {
    grad.fill(0.);
    let n = thetas.len();
    let (xs, ys) = coords.split_at(n);
    let (dxs, dys) = grad.split_at_mut(n);
    let mut fx = 0.;

    for i in 0..n {
        let (z, dp) = sd_polygon(&sums.contains[i], vec2(xs[i], ys[i]));
        let w = GAP - z;
        if w > 0. {
            fx += w * w;
            dxs[i] -= 2. * w * dp.x;
            dys[i] -= 2. * w * dp.y;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let (z, dp) = sd_polygon(
                sums.pairs[i][j].as_ref().unwrap(),
                vec2(xs[j], ys[j]) - vec2(xs[i], ys[i]),
            );
            if j == i + 1 {
                if z != 0. {
                    fx += z * z;
                    dxs[i] -= 2. * z * dp.x;
                    dys[i] -= 2. * z * dp.y;
                    dxs[j] += 2. * z * dp.x;
                    dys[j] += 2. * z * dp.y;
                }
            } else {
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
    }

    fx
}

fn optimize(
    sums: &Sums,
    mut monkeys: Monkeys,
    mut callback: impl FnMut(Option<&lbfgs::Info>, &[f64], &[f64]),
) -> (Monkeys, f64) {
    callback(None, &monkeys.thetas, &monkeys.coords);
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
        |coords, grad| val_and_grad(sums, &monkeys.thetas, coords, grad),
        &mut monkeys.coords,
    );
    callback(None, &monkeys.thetas, &monkeys.coords);
    let mut fx = f64::NAN;
    lbfgs::step_until(
        cfg,
        |coords, grad| val_and_grad(sums, &monkeys.thetas, coords, grad),
        &mut monkeys.coords,
        &mut state,
        |info| {
            callback(Some(&info), &monkeys.thetas, info.x);
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

fn arrangement(w: &mut impl fmt::Write, thetas: &[f64], coords: &[f64]) -> fmt::Result {
    let n = thetas.len();
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
            r##"  <use href="#monkey" transform="translate({} {}) rotate({})" />"##,
            coords[i],
            coords[n + i],
            thetas[i].to_degrees(),
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

fn get_sums(monkeys: &Monkeys) -> Sums {
    let contains: Vec<Polygon> = monkeys
        .thetas
        .iter()
        .map(|theta| {
            let sin = theta.sin();
            let cos = theta.cos();
            let monk: Polygon = monkey::POLYGON
                .iter()
                .map(|&v| {
                    let x = cos * v.x - sin * v.y;
                    let y = sin * v.x + cos * v.y;
                    -vec2(x, y)
                })
                .collect();
            minkowski_sum(&barrel::POLYGON, &monk, Orientation::Clockwise)
        })
        .collect();

    let pairs: Vec<Vec<Option<Polygon>>> = monkeys
        .thetas
        .iter()
        .enumerate()
        .map(|(i, theta1)| {
            let sin1 = theta1.sin();
            let cos1 = theta1.cos();
            let a: Polygon = monkey::POLYGON
                .iter()
                .map(|&v| {
                    let x = cos1 * v.x - sin1 * v.y;
                    let y = sin1 * v.x + cos1 * v.y;
                    vec2(x, y)
                })
                .collect();
            monkeys
                .thetas
                .iter()
                .enumerate()
                .map(|(j, theta2)| {
                    if j <= i {
                        return None;
                    }
                    let sin2 = theta2.sin();
                    let cos2 = theta2.cos();
                    let b: Polygon = monkey::POLYGON
                        .iter()
                        .map(|&v| {
                            let x = cos2 * v.x - sin2 * v.y;
                            let y = sin2 * v.x + cos2 * v.y;
                            -vec2(x, y)
                        })
                        .collect();
                    Some(minkowski_sum(&a, &b, Orientation::Counterclockwise))
                })
                .collect()
        })
        .collect();

    Sums { contains, pairs }
}

fn run(dir: &Path, sums: &Sums, monkeys: Monkeys) -> f64 {
    create_dir_all(dir).unwrap();
    let mut i: usize = 0;
    let (Monkeys { thetas, coords }, fx) = optimize(sums, monkeys, |info, thetas, coords| {
        if i.count_ones() < 2 {
            print!("i = {i}");
            if let Some(info) = info {
                println!(", fx = {}", info.fx);
            } else {
                println!();
            }
            let mut s = String::new();
            arrangement(&mut s, thetas, coords).unwrap();
            File::create(dir.join(format!("{i}.svg")))
                .unwrap()
                .write_all(s.as_bytes())
                .unwrap();
            rasterize(&s)
                .save_png(dir.join(format!("{i}.png")))
                .unwrap();
        }
        i += 1;
    });
    i -= 1;
    println!("i = {i}, fx = {fx}");
    let mut s = String::new();
    arrangement(&mut s, &thetas, &coords).unwrap();
    File::create(dir.join(format!("{i}.svg")))
        .unwrap()
        .write_all(s.as_bytes())
        .unwrap();
    rasterize(&s)
        .save_png(dir.join(format!("{i}.png")))
        .unwrap();
    fx
}

fn main() {
    assert_eq!(orientation(&barrel::POLYGON), Orientation::Clockwise);
    assert_eq!(orientation(&monkey::POLYGON), Orientation::Counterclockwise);
    let dir = Path::new("out");
    let n = 6;
    let seed = 0;
    let monkeys = init(seed, n);
    let sums = get_sums(&monkeys);
    run(&dir.join(format!("{n}-{seed}")), &sums, monkeys);
}
