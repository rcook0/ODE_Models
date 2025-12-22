
use wasm_bindgen::prelude::*;
use ode_solvers::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Series {
    pub t: Vec<f64>,
    pub y: Vec<f64>, // flattened
    pub dim: usize,
}

fn flatten<const N: usize>(ys: &Vec<[f64; N]>) -> Vec<f64> {
    ys.iter().flat_map(|v| v.iter().cloned()).collect()
}

struct Lin2 { a11:f64,a12:f64,a21:f64,a22:f64 }
impl System<[f64;2]> for Lin2 {
    fn system(&self,_t:f64,y:&[f64;2],dy:&mut[f64;2]){
        dy[0]=self.a11*y[0]+self.a12*y[1];
        dy[1]=self.a21*y[0]+self.a22*y[1];
    }
}
#[wasm_bindgen]
pub fn integrate_linear_2x2(a11:f64,a12:f64,a21:f64,a22:f64,x0:f64,y0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Lin2{a11,a12,a21,a22};
    let mut y=[x0,y0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:2 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Vdp{ mu:f64 }
impl System<[f64;2]> for Vdp {
    fn system(&self,_t:f64,y:&[f64;2],dy:&mut[f64;2]){
        let x=y[0]; let v=y[1];
        dy[0]=v;
        dy[1]=self.mu*(1.0-x*x)*v - x;
    }
}
#[wasm_bindgen]
pub fn integrate_vanderpol(mu:f64,x0:f64,v0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Vdp{mu};
    let mut y=[x0,v0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:2 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct LV{ alpha:f64,beta:f64,delta:f64,gamma:f64 }
impl System<[f64;2]> for LV {
    fn system(&self,_t:f64,y:&[f64;2],dy:&mut[f64;2]){
        let x=y[0]; let p=y[1];
        dy[0]=self.alpha*x - self.beta*x*p;
        dy[1]=self.delta*x*p - self.gamma*p;
    }
}
#[wasm_bindgen]
pub fn integrate_predator_prey(alpha:f64,beta:f64,delta:f64,gamma:f64,x0:f64,y0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=LV{alpha,beta,delta,gamma};
    let mut y=[x0,y0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:2 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Duff{ delta:f64,alpha:f64,beta:f64,gamma:f64,omega:f64 }
impl System<[f64;2]> for Duff {
    fn system(&self,t:f64,y:&[f64;2],dy:&mut[f64;2]){
        let x=y[0]; let v=y[1];
        dy[0]=v;
        dy[1]=-self.delta*v - self.alpha*x - self.beta*x*x*x + self.gamma*(self.omega*t).cos();
    }
}
#[wasm_bindgen]
pub fn integrate_duffing(delta:f64,alpha:f64,beta:f64,gamma:f64,omega:f64,x0:f64,v0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Duff{delta,alpha,beta,gamma,omega};
    let mut y=[x0,v0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:2 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Sir{ beta:f64,gamma:f64 }
impl System<[f64;3]> for Sir {
    fn system(&self,_t:f64,y:&[f64;3],dy:&mut[f64;3]){
        let s=y[0]; let i=y[1]; let r=y[2];
        let n=s+i+r;
        dy[0] = -self.beta*s*i/(n+1e-12);
        dy[1] =  self.beta*s*i/(n+1e-12) - self.gamma*i;
        dy[2] =  self.gamma*i;
    }
}
#[wasm_bindgen]
pub fn integrate_sir(beta:f64,gamma:f64,s0:f64,i0:f64,r0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Sir{beta,gamma};
    let mut y=[s0,i0,r0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Lorenz{ sigma:f64,rho:f64,beta:f64 }
impl System<[f64;3]> for Lorenz {
    fn system(&self,_t:f64,y:&[f64;3],dy:&mut[f64;3]){
        let x=y[0]; let yy=y[1]; let z=y[2];
        dy[0]=self.sigma*(yy-x);
        dy[1]=x*(self.rho - z) - yy;
        dy[2]=x*yy - self.beta*z;
    }
}
#[wasm_bindgen]
pub fn integrate_lorenz63(sigma:f64,rho:f64,beta:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Lorenz{sigma,rho,beta};
    let mut y=[x0,y0,z0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Rossler{ a:f64,b:f64,c:f64 }
impl System<[f64;3]> for Rossler {
    fn system(&self,_t:f64,y:&[f64;3],dy:&mut[f64;3]){
        let x=y[0]; let yy=y[1]; let z=y[2];
        dy[0] = -yy - z;
        dy[1] = x + self.a*yy;
        dy[2] = self.b + z*(x - self.c);
    }
}
#[wasm_bindgen]
pub fn integrate_rossler(a:f64,b:f64,c:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Rossler{a,b,c};
    let mut y=[x0,y0,z0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}
