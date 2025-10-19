use wasm_bindgen::prelude::*;
use ode_solvers::*;
use serde::{Serialize, Deserialize};
#[derive(Serialize, Deserialize)]
pub struct Series{pub t:Vec<f64>,pub y:Vec<f64>,pub dim:usize}
struct Lin2{a11:f64,a12:f64,a21:f64,a22:f64}
impl System<[f64;2]> for Lin2{fn system(&self,_t:f64,y:&[f64;2],dy:&mut[f64;2]){dy[0]=self.a11*y[0]+self.a12*y[1];dy[1]=self.a21*y[0]+self.a22*y[1];}}
#[wasm_bindgen]
pub fn integrate_linear_2x2(a11:f64,a12:f64,a21:f64,a22:f64,x0:f64,y0:f64,t0:f64,t1:f64,dt:f64)->JsValue{let sys=Lin2{a11,a12,a21,a22};let mut y=[x0,y0];let mut stepper=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);let _=stepper.integrate();let ts=stepper.x_out().to_vec();let ys=stepper.y_out().iter().flat_map(|v|v.iter().cloned()).collect::<Vec<f64>>();serde_wasm_bindgen::to_value(&Series{t:ts,y:ys,dim:2}).unwrap()}
struct Vdp{mu:f64}
impl System<[f64;2]> for Vdp{fn system(&self,_t:f64,y:&[f64;2],dy:&mut[f64;2]){let x=y[0];let v=y[1];dy[0]=v;dy[1]=self.mu*(1.0-x*x)*v-x;}}
#[wasm_bindgen]
pub fn integrate_vanderpol(mu:f64,x0:f64,v0:f64,t0:f64,t1:f64,dt:f64)->JsValue{let sys=Vdp{mu};let mut y=[x0,v0];let mut stepper=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);let _=stepper.integrate();let ts=stepper.x_out().to_vec();let ys=stepper.y_out().iter().flat_map(|v|v.iter().cloned()).collect::<Vec<f64>>();serde_wasm_bindgen::to_value(&Series{t:ts,y:ys,dim:2}).unwrap()}