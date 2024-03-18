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
use util::{dot, vec2, Vec2};

const BARREL_BEZIER: &str = include_str!("BarrelCubicBézier.txt");
const MONKEY_BEZIER: &str = include_str!("MonkeyCubicBézier.txt");

const WIDTH: f64 = 1554.197998;
const HEIGHT: f64 = 2507.9672852;
const GAP: f64 = 3.;

type Polygon = Vec<Vec2>;

// https://iquilezles.org/articles/distfunctions2d/
fn sd_polygon(v: &[Vec2], p: Vec2) -> (f64, Vec2) {
    let n = v.len();
    let u = p - v[0];
    let mut d = dot(u, u);
    let mut dp = 2. * u;
    let mut s = 1.0;
    let mut i = 0;
    let mut j = n - 1;
    while i < n {
        let e = v[j] - v[i];
        let w = p - v[i];
        let we = dot(w, e);
        let ee = dot(e, e);
        let r = we / ee;
        let rc = r.clamp(0.0, 1.0);
        let b = w - e * rc;
        let bb = dot(b, b);
        if bb < d {
            d = bb;
            let db = 2. * b;
            let drc = -dot(e, db);
            let dr = if (0.0..=1.0).contains(&r) { drc } else { 0. };
            let dwe = dr / ee;
            let dw = db + dwe * e;
            dp = dw;
        }
        let c = [p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x];
        if c.iter().all(|&a| a) || c.iter().all(|&a| !a) {
            s *= -1.0;
        }
        j = i;
        i += 1;
    }
    let z = s * d.sqrt();
    (z, dp / (2. * z))
}

struct Monkeys {
    coords: Vec<f64>,
}

fn init(n: usize, seed: u64) -> Monkeys {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut coords: Vec<_> = (0..n).map(|_| rng.gen_range(0.0..WIDTH)).collect();
    coords.extend((0..n).map(|_| rng.gen_range(0.0..HEIGHT)));
    Monkeys { coords }
}

fn val_and_grad(coords: &[f64], grad: &mut [f64]) -> f64 {
    grad.fill(0.);
    let n = coords.len() / 2;
    let (x, y) = coords.split_at(n);
    let (dx, dy) = grad.split_at_mut(n);
    let mut fx = 0.;

    let barr: Vec<Point> = barrel::POLYGON
        .into_iter()
        .map(|Vec2 { x, y }| (x, y))
        .collect();
    // unclear why `barr` doesn't need to be reversed here
    for i in 0..n {
        let monk: Vec<Point> = monkey::POLYGON
            .iter()
            .map(|Vec2 { x, y }| (-x, -y))
            .collect();
        // unclear why `monk` doesn't need to be reversed here
        let sum: Polygon = extract_loops(&reduced_convolution(&barr, &monk))
            .swap_remove(0)
            .into_iter()
            .map(|((x, y), _)| Vec2 { x, y })
            .collect();
        let (z, dp) = sd_polygon(&sum, vec2(x[i], y[i]));
        let w = z + GAP;
        if w > 0. {
            fx += w * w;
            dx[i] += 2. * w * dp.x;
            dy[i] += 2. * w * dp.y;
        }
    }

    // for i in 0..n {
    //     for j in (i + 1)..n {
    //         let (z, dp) = sd_polygon(
    //             &sums.pairs[indices[i]][indices[j]],
    //             (vec2(x[j], y[j]) - vec2(x[i], y[i])) / SCALE,
    //         );
    //         let w = GAP - SCALE * z;
    //         if w > 0. {
    //             fx += w * w;
    //             dx[i] += 2. * w * dp.x;
    //             dy[i] += 2. * w * dp.y;
    //             dx[j] -= 2. * w * dp.x;
    //             dy[j] -= 2. * w * dp.y;
    //         }
    //     }
    // }

    fx
}

fn optimize(
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
    let mut state = lbfgs::first_step(cfg, val_and_grad, &mut monkeys.coords);
    callback(None, &monkeys.coords);
    let mut fx = f64::NAN;
    lbfgs::step_until(cfg, val_and_grad, &mut monkeys.coords, &mut state, |info| {
        callback(Some(&info), info.x);
        if info.fx == fx {
            Some(())
        } else {
            fx = info.fx;
            None
        }
    });
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
    let (Monkeys { coords }, fx) = optimize(init(n, seed), |info, coords| {
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
    let dir = Path::new("out");
    run(dir, 10, 0);
}
