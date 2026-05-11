use libtt_metal_cxx::Program;

fn main() {
    let mut program = Program::create();
    println!("runtime_id={:?}", program.runtime_id());

    program.set_runtime_id(1);
    println!("updated_runtime_id={:?}", program.runtime_id());
}
