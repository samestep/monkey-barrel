use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

pub const fn vec2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Vec2 {
        vec2(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Vec2 {
        vec2(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Vec2 {
        vec2(self * rhs.x, self * rhs.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f64) -> Vec2 {
        vec2(self.x * rhs, self.y * rhs)
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f64) -> Vec2 {
        vec2(self.x / rhs, self.y / rhs)
    }
}

pub fn dot(u: Vec2, v: Vec2) -> f64 {
    u.x * v.x + u.y * v.y
}
