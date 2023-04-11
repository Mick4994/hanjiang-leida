import serial
import numpy as np
from crccheck.crc import Crc8, Crc16

class SerialSender:
    def __init__(self, com = "COM4") -> None:
        self.crc8 = Crc8()
        self.crc16 = Crc16()
        self.com = com

        sof = b'\xa5' # 数据帧起始字节，固定值为0xA5
        data_length = b'\x0d' # 数据帧中data的长度
        seq = b'\x01' # 包序号
        frame_header = sof + data_length + seq
        frame_header += bytes([self.crc8.calc(frame_header)])

        cmd_id = b'\x03\x03'

        self.head_data = frame_header + cmd_id

    def Send(self, xcnp_map):
        for id, min_point_3d in xcnp_map:
            id = bytes([id])
            xyz_bytes = np.array(min_point_3d, dtype=np.float32).tobytes()
            data_pack = xyz_bytes + b"\x00" + id
            # 小地图交互数据
            send_data = self.head_data + data_pack

            # crc16校验整包
            crc16_out = self.crc16.calc(send_data)
            high_crc16 = crc16_out // 256
            low_crc16 = crc16_out - high_crc16 * 256
            send_data += bytes([high_crc16])
            send_data += bytes([low_crc16])

            self.send_data = send_data

            with serial.Serial(port=self.com, 
                        baudrate=115200, 
                        bytesize=8, 
                        stopbits=1) as my_serial:
                my_serial.write(send_data)

if __name__ == "__main__":

    crc8 = Crc8()
    crc16 = Crc16()

    sof = b'\xa5' # 数据帧起始字节，固定值为0xA5
    data_length = b'\x0d' # 数据帧中data的长度
    seq = b'\x01' # 包序号
    frame_header = sof + data_length + seq
    frame_header += bytes([crc8.calc(frame_header)])

    cmd_id = b'\x03\x03'

    data = b'\x00'
    for i in range(12):
        data += b'\x00'
    
    send_data = frame_header + cmd_id + data

    crc16_out = crc16.calc(send_data)

    high_crc16 = crc16_out // 256
    low_crc16 = crc16_out - high_crc16 * 256


    send_data += bytes([high_crc16])
    send_data += bytes([low_crc16])


    with serial.Serial(port="COM4", 
                       baudrate=115200, 
                       bytesize=8, 
                       stopbits=1) as my_serial:
        my_serial.write(send_data)

    print(send_data.hex())


    