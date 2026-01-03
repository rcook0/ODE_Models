
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


// ---------- Simple adaptive RK45 (Cash–Karp) for 3D systems ----------
fn rk45_step_3(f: &dyn Fn(f64, [f64;3]) -> [f64;3], t: f64, y: [f64;3], h: f64) -> ([f64;3], [f64;3]) {
    let a2=0.2;
    let a3=0.3;
    let a4=0.6;
    let a5=1.0;
    let a6=0.875;

    let b21=0.2;

    let b31=3.0/40.0; let b32=9.0/40.0;

    let b41=0.3; let b42=-0.9; let b43=1.2;

    let b51=-11.0/54.0; let b52=2.5; let b53=-70.0/27.0; let b54=35.0/27.0;

    let b61=1631.0/55296.0; let b62=175.0/512.0; let b63=575.0/13824.0; let b64=44275.0/110592.0; let b65=253.0/4096.0;

    let c1=37.0/378.0; let c3=250.0/621.0; let c4=125.0/594.0; let c6=512.0/1771.0;
    let dc1=c1 - 2825.0/27648.0;
    let dc3=c3 - 18575.0/48384.0;
    let dc4=c4 - 13525.0/55296.0;
    let dc5= -277.0/14336.0;
    let dc6=c6 - 0.25;

    let k1 = f(t, y);

    let y2 = [y[0]+h*b21*k1[0], y[1]+h*b21*k1[1], y[2]+h*b21*k1[2]];
    let k2 = f(t + a2*h, y2);

    let y3 = [y[0]+h*(b31*k1[0]+b32*k2[0]), y[1]+h*(b31*k1[1]+b32*k2[1]), y[2]+h*(b31*k1[2]+b32*k2[2])];
    let k3 = f(t + a3*h, y3);

    let y4 = [y[0]+h*(b41*k1[0]+b42*k2[0]+b43*k3[0]), y[1]+h*(b41*k1[1]+b42*k2[1]+b43*k3[1]), y[2]+h*(b41*k1[2]+b42*k2[2]+b43*k3[2])];
    let k4 = f(t + a4*h, y4);

    let y5 = [y[0]+h*(b51*k1[0]+b52*k2[0]+b53*k3[0]+b54*k4[0]), y[1]+h*(b51*k1[1]+b52*k2[1]+b53*k3[1]+b54*k4[1]), y[2]+h*(b51*k1[2]+b52*k2[2]+b53*k3[2]+b54*k4[2])];
    let k5 = f(t + a5*h, y5);

    let y6 = [y[0]+h*(b61*k1[0]+b62*k2[0]+b63*k3[0]+b64*k4[0]+b65*k5[0]),
              y[1]+h*(b61*k1[1]+b62*k2[1]+b63*k3[1]+b64*k4[1]+b65*k5[1]),
              y[2]+h*(b61*k1[2]+b62*k2[2]+b63*k3[2]+b64*k4[2]+b65*k5[2])];
    let k6 = f(t + a6*h, y6);

    let yout = [
        y[0] + h*(c1*k1[0] + c3*k3[0] + c4*k4[0] + c6*k6[0]),
        y[1] + h*(c1*k1[1] + c3*k3[1] + c4*k4[1] + c6*k6[1]),
        y[2] + h*(c1*k1[2] + c3*k3[2] + c4*k4[2] + c6*k6[2]),
    ];

    let err = [
        h*(dc1*k1[0] + dc3*k3[0] + dc4*k4[0] + dc5*k5[0] + dc6*k6[0]),
        h*(dc1*k1[1] + dc3*k3[1] + dc4*k4[1] + dc5*k5[1] + dc6*k6[1]),
        h*(dc1*k1[2] + dc3*k3[2] + dc4*k4[2] + dc5*k5[2] + dc6*k6[2]),
    ];
    (yout, err)
}

fn integrate_adaptive_3(
    f: &dyn Fn(f64, [f64;3]) -> [f64;3],
    t0: f64, t1: f64, mut y: [f64;3],
    dt0: f64, rtol: f64, atol: f64, max_steps: usize
) -> (Vec<f64>, Vec<[f64;3]>) {
    let mut t = t0;
    let mut h = dt0.abs().max(1e-6);
    let mut ts = Vec::new();
    let mut ys = Vec::new();
    ts.push(t); ys.push(y);

    for _ in 0..max_steps {
        if t >= t1 { break; }
        if t + h > t1 { h = t1 - t; }
        let (ynew, err) = rk45_step_3(f, t, y, h);
        let sc0 = atol + rtol * ynew[0].abs().max(y[0].abs());
        let sc1 = atol + rtol * ynew[1].abs().max(y[1].abs());
        let sc2 = atol + rtol * ynew[2].abs().max(y[2].abs());
        let e = ((err[0]/sc0).powi(2) + (err[1]/sc1).powi(2) + (err[2]/sc2).powi(2)).sqrt() / 3.0_f64.sqrt();

        if e <= 1.0 {
            t += h;
            y = ynew;
            ts.push(t);
            ys.push(y);
        }
        let safety = 0.9;
        let pow = 0.2;
        let factor = if e == 0.0 { 5.0 } else { (safety * e.powf(-pow)).clamp(0.2, 5.0) };
        h = (h * factor).clamp(1e-6, 1.0);
    }
    (ts, ys)
}

#[wasm_bindgen]
pub fn integrate_lorenz63_adaptive(sigma:f64,rho:f64,beta:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt0:f64,rtol:f64,atol:f64,max_steps:usize)->JsValue{
    let f = |_:f64, y:[f64;3]| -> [f64;3] {
        let x=y[0]; let yy=y[1]; let z=y[2];
        [sigma*(yy-x), x*(rho-z)-yy, x*yy - beta*z]
    };
    let (ts, ys) = integrate_adaptive_3(&f, t0, t1, [x0,y0,z0], dt0, rtol, atol, max_steps);
    let series = Series{ t: ts, y: flatten(&ys), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

#[wasm_bindgen]
pub fn integrate_rossler_adaptive(a:f64,b:f64,c:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt0:f64,rtol:f64,atol:f64,max_steps:usize)->JsValue{
    let f = |_:f64, y:[f64;3]| -> [f64;3] {
        let x=y[0]; let yy=y[1]; let z=y[2];
        [-yy - z, x + a*yy, b + z*(x - c)]
    };
    let (ts, ys) = integrate_adaptive_3(&f, t0, t1, [x0,y0,z0], dt0, rtol, atol, max_steps);
    let series = Series{ t: ts, y: flatten(&ys), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

// ---------- Extra advanced systems (fixed-step) ----------
struct Chua{ alpha:f64,beta:f64,m0:f64,m1:f64 }
impl System<[f64;3]> for Chua {
    fn system(&self,_t:f64,y:&[f64;3],dy:&mut[f64;3]){
        let x=y[0]; let yy=y[1]; let z=y[2];
        let fx = self.m1*x + 0.5*(self.m0-self.m1)*((x+1.0).abs() - (x-1.0).abs());
        dy[0]=self.alpha*(yy - x - fx);
        dy[1]=x - yy + z;
        dy[2]=-self.beta*yy;
    }
}
#[wasm_bindgen]
pub fn integrate_chua(alpha:f64,beta:f64,m0:f64,m1:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Chua{alpha,beta,m0,m1};
    let mut y=[x0,y0,z0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}

struct Thomas{ a:f64 }
impl System<[f64;3]> for Thomas {
    fn system(&self,_t:f64,y:&[f64;3],dy:&mut[f64;3]){
        let x=y[0]; let yy=y[1]; let z=y[2];
        dy[0]=yy.sin() - self.a*x;
        dy[1]=z.sin()  - self.a*yy;
        dy[2]=x.sin()  - self.a*z;
    }
}
#[wasm_bindgen]
pub fn integrate_thomas(a:f64,x0:f64,y0:f64,z0:f64,t0:f64,t1:f64,dt:f64)->JsValue{
    let sys=Thomas{a};
    let mut y=[x0,y0,z0];
    let mut s=Dopri5::new(sys,t0,t1,dt,&mut y,1e-6,1e-9);
    let _=s.integrate();
    let series=Series{ t:s.x_out().to_vec(), y:flatten(s.y_out()), dim:3 };
    serde_wasm_bindgen::to_value(&series).unwrap()
}
