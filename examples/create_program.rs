use libtt_metal_cxx::Program;

fn main() {
    let mut program = Program::new();
    println!("runtime_id={:?}", program.runtime_id());

    program.set_runtime_id(1);
    println!("updated_runtime_id={:?}", program.runtime_id());
}
