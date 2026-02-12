from tdmclient import ClientAsync
from tdmclient.atranspiler import ATranspiler

thymio_program_python = r"""
@onevent
def prox():
    global prox_horizontal, motor_left_target, motor_right_target
    prox_front = prox_horizontal[2]
    speed = -prox_front // 10
    motor_left_target = speed
    motor_right_target = speed
"""

# convert program from Python to Aseba
thymio_program_aseba = ATranspiler.simple_transpile(thymio_program_python)

with ClientAsync() as client:
    async def prog():
        with await client.lock() as node:
            error = await node.compile(thymio_program_aseba)
            error = await node.run()

    client.run_async_program(prog)
