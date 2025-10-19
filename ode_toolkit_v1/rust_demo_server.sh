cd rust-wasm && wasm-pack build --target web --release
cd .. && cp -r rust-wasm/pkg web/
python3 -m http.server -d web 8080
# open http://localhost:8080 and press “Run”
