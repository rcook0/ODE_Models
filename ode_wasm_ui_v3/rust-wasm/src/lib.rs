use wasm_bindgen::prelude::*;
use ode_solvers::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Series {
    pub t: Vec<f64>,
    pub y: Vec<f64>,    // flattened: n_steps * dim
    pub dim: usize
}

// ---------- Linear 2x2: y' = A y ----------
struct Lin2 { a11: f64, a12: f64, a21: f64, a22: f64 }
impl System<[f64;2]> for Lin2 {
    fn system(&self, _t: f64, y: &[f64;2], dy: &mut [f64;2]) {
        dy[0] = self.a11*y[0] + self.a12*y[1];
        dy[1] = self.a21*y[0] + self.a22*y[1];
    }
}
#[wasm_bindgen]
pub fn integrate_linear_2x2(a11:f64,a12:f64,a21:f64,a22:f64,
                             x0:f64,y0:f64,
                             t0:f64,t1:f64,dt:f64) -> JsValue {
    let sys = Lin2{a11,a12,a21,a22};
    let mut y = [x0,y0];
    let mut stepper = Dopri5::new(sys, t0, t1, dt, &mut y, 1e-6, 1e-9);
    let _ = stepper.integrate();
    serde_wasm_bindgen::to_value(&Series{
        t: stepper.x_out().to_vec(),
        y: stepper.y_out().iter().flat_map(|v| v.iter().cloned()).collect::<Vec<f64>>(),
        dim: 2
    }).unwrap()
}

// ---------- Van der Pol ----------
struct Vdp { mu: f64 }
impl System<[f64;2]> for Vdp {
    fn system(&self, _t: f64, y: &[f64;2], dy: &mut [f64;2]) {
        let x = y[0];
        let v = y[1];
        dy[0] = v;
        dy[1] = self.mu*(1.0 - x*x)*v - x;
    }
}
#[wasm_bindgen]
pub fn integrate_vanderpol(mu:f64, x0:f64,v0:f64,t0:f64,t1:f64,dt:f64) -> JsValue {
    let sys = Vdp{mu};
    let mut y = [x0,v0];
    let mut stepper = Dopri5::new(sys, t0, t1, dt, &mut y, 1e-6, 1e-9);
    let _ = stepper.integrate();
    serde_wasm_bindgen::to_value(&Series{
        t: stepper.x_out().to_vec(),
        y: stepper.y_out().iter().flat_map(|v| v.iter().cloned()).collect::<Vec<f64>>(),
        dim: 2
    }).unwrap()
}

// ---------- Predator–Prey (Lotka–Volterra) ----------
struct LV { alpha: f64, beta: f64, delta: f64, gamma: f64 }
impl System<[f64;2]> for LV {
    fn system(&self, _t: f64, y: &[f64;2], dy: &mut [f64;2]) {
        let x = y[0];   // prey
        let p = y[1];   // predator
        dy[0] = self.alpha*x - self.beta*x*p;
        dy[1] = self.delta*x*p - self.gamma*p;
    }
}
#[wasm_bindgen]
pub fn integrate_predator_prey(alpha:f64,beta:f64,delta:f64,gamma:f64,
                               x0:f64,p0:f64,t0:f64,t1:f64,dt:f64) -> JsValue {
    let sys = LV{alpha,beta,delta,gamma};
    let mut y = [x0,p0];
    let mut stepper = Dopri5::new(sys, t0, t1, dt, &mut y, 1e-6, 1e-9);
    let _ = stepper.integrate();
    serde_wasm_bindgen::to_value(&Series{
        t: stepper.x_out().to_vec(),
        y: stepper.y_out().iter().flat_map(|v| v.iter().cloned()).collect::<Vec<f64>>(),
        dim: 2
    }).unwrap()
}

// ---------- Duffing Oscillator (driven, damped) ----------
struct Duff { delta: f64, alpha: f64, beta: f64, gamma: f64, omega: f64 }
impl System<[f64;2]> for Duff {
    fn system(&self, t: f64, y: &[f64;2], dy: &mut [f64;2]) {
        let x = y[0];
        let v = y[1];
        dy[0] = v;
        dy[1] = -self.delta*v - self.alpha*x - self.beta*x*x*x + self.gamma*(self.omega*t).cos();
    }
}
#[wasm_bindgen]
pub fn integrate_duffing(delta:f64,alpha:f64,beta:f64,gamma:f64,omega:f64,
                         x0:f64,v0:f64,t0:f64,t1:f64,dt:f64) -> JsValue {
    let sys = Duff{delta,alpha,beta,gamma,omega};
    let mut y = [x0,v0];
    let mut stepper = Dopri5::new(sys, t0, t1, dt, &mut y, 1e-6, 1e-9);
    let _ = stepper.integrate();
    serde_wasm_bindgen::to_value(&Series{
        t: stepper.x_out().to_vec(),
        y: stepper.y_out().iter().flat_map(|v| v.iter().cloned()).collect::<Vec<f64>>(),
        dim: 2
    }).unwrap()
}
