mod barrel;
mod lbfgs;
mod monkey;
mod util;

use minkowski::{extract_loops, reduced_convolution, Point, Pseudovert};
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

const EPS: f64 = 1e-6;

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

fn center(p: &[Vec2]) -> Vec2 {
    let mut sum = vec2(0., 0.);
    for &v in p {
        sum = sum + v;
    }
    sum / p.len() as f64
}

struct PointGrad {
    i: usize,
    g: Vec2,
}

fn sd_polygon(q: &[Vec2], p: Vec2) -> (f64, PointGrad, PointGrad) {
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
        let d = cross(u, v) / z;
        let gu = vec2(v.y, -v.x) / z;
        let gv = (vec2(-u.y, u.x) - (d / z) * v) / z;
        (d, PointGrad { i, g: -gu - gv }, PointGrad { i: j, g: gv })
    } else {
        let d = dd.sqrt();
        let gi = u / d;
        let gj = PointGrad {
            i: j,
            g: vec2(0., 0.),
        };
        if s {
            (-d, PointGrad { i, g: gi }, gj)
        } else {
            (d, PointGrad { i, g: -gi }, gj)
        }
    }
}

fn vecify((x, y): Point) -> Vec2 {
    vec2(x, y)
}

fn pointize(p: &[Vec2]) -> Vec<Point> {
    p.iter().map(|&Vec2 { x, y }| (x, y)).collect()
}

fn sd_minkowski_sum(
    a: &[Vec2],
    b: &[Vec2],
    o: Orientation,
) -> (f64, Vec<PointGrad>, Vec<PointGrad>) {
    let edges = reduced_convolution(&pointize(a), &pointize(b));
    let mut loops = extract_loops(&edges);
    loops.sort_by_key(|p| p.len()); // TODO: actually compute nesting instead of this hack
    let c = loops.pop().unwrap();
    let poly: Polygon = c.iter().map(|&((x, y), _)| vec2(x, y)).collect();
    assert_eq!(orientation(&poly), o);
    let (z, g1, g2) = sd_polygon(&poly, vec2(0., 0.));
    let mut das = vec![];
    let mut dbs = vec![];
    for PointGrad { i: k, g } in [g1, g2] {
        match c[k] {
            (_, Pseudovert::Given { i, j }) => {
                das.push(PointGrad { i, g });
                dbs.push(PointGrad { i: j, g });
            }
            (_, Pseudovert::Steiner { m, n }) => {
                let a0 = vecify(edges[m].p.z);
                let a1 = vecify(edges[m].q.z);
                let u = a1 - a0;
                let b0 = vecify(edges[n].p.z);
                let b1 = vecify(edges[n].q.z);
                let v = b1 - b0;
                let denom = cross(u, v);
                let a = cross(a0, a1);
                let b = cross(b0, b1);
                let w = vec2(a, b);
                let xs = vec2(u.x, v.x);
                let ys = vec2(u.y, v.y);
                let x = cross(xs, w);
                let y = cross(ys, w);
                let p = vec2(x, y) / denom;
                let dy = g.y / denom;
                let dx = g.x / denom;
                let ddenom = -(dy * p.y + dx * p.x);
                let dys = dy * vec2(w.y, -w.x);
                let dxs = dx * vec2(w.y, -w.x);
                let dw = dy * vec2(-ys.y, ys.x) + dx * vec2(-xs.y, xs.x);
                let db = dw.y;
                let da = dw.x;
                let dv = vec2(dxs.y, dys.y) + ddenom * vec2(-u.y, u.x);
                let du = vec2(dxs.x, dys.x) + ddenom * vec2(v.y, -v.x);
                let db1 = db * vec2(-b0.y, b0.x) + dv;
                let db0 = db * vec2(b1.y, -b1.x) - dv;
                let da1 = da * vec2(-a0.y, a0.x) + du;
                let da0 = da * vec2(a1.y, -a1.x) - du;
                das.push(PointGrad {
                    i: edges[m].p.i,
                    g: da0,
                });
                das.push(PointGrad {
                    i: edges[m].q.i,
                    g: da1,
                });
                das.push(PointGrad {
                    i: edges[n].p.i,
                    g: db0,
                });
                das.push(PointGrad {
                    i: edges[n].q.i,
                    g: db1,
                });
                dbs.push(PointGrad {
                    i: edges[m].p.j,
                    g: da0,
                });
                dbs.push(PointGrad {
                    i: edges[m].q.j,
                    g: da1,
                });
                dbs.push(PointGrad {
                    i: edges[n].p.j,
                    g: db0,
                });
                dbs.push(PointGrad {
                    i: edges[n].q.j,
                    g: db1,
                });
            }
        }
    }
    (z, das, dbs)
}

struct Monkeys {
    coords: Vec<f64>,
}

fn init(rng: &mut impl Rng, n: usize) -> Monkeys {
    let mut coords: Vec<_> = (0..n).map(|_| rng.gen_range(0.0..WIDTH)).collect();
    coords.extend((0..n).map(|_| rng.gen_range(0.0..HEIGHT)));
    coords.extend((0..n).map(|_| rng.gen_range(0.0..TAU)));
    Monkeys { coords }
}

fn val_and_grad(rng: &mut impl Rng, coords: &[f64], grad: &mut [f64]) -> f64 {
    grad.fill(0.);
    let n = coords.len() / 3;
    let (xs, others) = coords.split_at(n);
    let (ys, thetas) = others.split_at(n);
    let (dxs, dothers) = grad.split_at_mut(n);
    let (dys, dthetas) = dothers.split_at_mut(n);
    let mut fx = 0.;

    let monkey_center = center(&monkey::POLYGON);

    let barr: Polygon = barrel::POLYGON
        .into_iter()
        .map(|v| v + vec2(rng.gen_range(-EPS..EPS), rng.gen_range(-EPS..EPS)))
        .collect();
    for i in 0..n {
        let theta = thetas[i];
        let sin = theta.sin();
        let cos = theta.cos();
        let monk: Polygon = monkey::POLYGON
            .iter()
            .map(|&u| {
                let v = u - monkey_center;
                let x = cos * v.x - sin * v.y;
                let y = sin * v.x + cos * v.y;
                let w = vec2(xs[i], ys[i]) + vec2(x, y) + monkey_center;
                -(w + vec2(rng.gen_range(-EPS..EPS), rng.gen_range(-EPS..EPS)))
            })
            .collect();
        let (z, _, dmonk) = sd_minkowski_sum(&barr, &monk, Orientation::Clockwise);
        let w = GAP - z;
        if w > 0. {
            fx += w * w;
            for PointGrad { i: j, g } in dmonk {
                dxs[i] += 2. * w * g.x;
                dys[i] += 2. * w * g.y;
                let v = monkey::POLYGON[j] - monkey_center;
                let dsin = cross(v, g);
                let dcos = dot(v, g);
                dthetas[i] += 2. * w * (dsin * cos - dcos * sin);
            }
        }
    }

    for i in 0..n {
        let thetai = thetas[i];
        let sini = thetai.sin();
        let cosi = thetai.cos();
        for j in (i + 1)..n {
            let a: Polygon = monkey::POLYGON
                .iter()
                .map(|&u| {
                    let v = u - monkey_center;
                    let x = cosi * v.x - sini * v.y;
                    let y = sini * v.x + cosi * v.y;
                    let w = vec2(xs[i], ys[i]) + vec2(x, y) + monkey_center;
                    w + vec2(rng.gen_range(-EPS..EPS), rng.gen_range(-EPS..EPS))
                })
                .collect();

            let thetaj = thetas[j];
            let sinj = thetaj.sin();
            let cosj = thetaj.cos();
            let b: Polygon = monkey::POLYGON
                .iter()
                .map(|&u| {
                    let v = u - monkey_center;
                    let x = cosj * v.x - sinj * v.y;
                    let y = sinj * v.x + cosj * v.y;
                    let w = vec2(xs[j], ys[j]) + vec2(x, y) + monkey_center;
                    -(w + vec2(rng.gen_range(-EPS..EPS), rng.gen_range(-EPS..EPS)))
                })
                .collect();

            let (z, das, dbs) = sd_minkowski_sum(&a, &b, Orientation::Counterclockwise);
            let w = GAP - z;
            if w > 0. {
                fx += w * w;
                for PointGrad { i: k, g } in das {
                    dxs[i] -= 2. * w * g.x;
                    dys[i] -= 2. * w * g.y;
                    let v = monkey::POLYGON[k] - monkey_center;
                    let dsin = cross(v, g);
                    let dcos = dot(v, g);
                    dthetas[i] -= 2. * w * (dsin * cosi - dcos * sini);
                }
                for PointGrad { i: k, g } in dbs {
                    dxs[j] += 2. * w * g.x;
                    dys[j] += 2. * w * g.y;
                    let v = monkey::POLYGON[k] - monkey_center;
                    let dsin = cross(v, g);
                    let dcos = dot(v, g);
                    dthetas[j] += 2. * w * (dsin * cosj - dcos * sinj);
                }
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
    let n = coords.len() / 3;
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
    let monkey_center = center(&monkey::POLYGON);
    for i in 0..n {
        writeln!(
            w,
            r##"  <use href="#monkey" transform="translate({} {}) rotate({} {} {})" />"##,
            coords[i],
            coords[n + i],
            coords[2 * n + i].to_degrees(),
            monkey_center.x,
            monkey_center.y,
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
    run(dir, 10, 0);
}
