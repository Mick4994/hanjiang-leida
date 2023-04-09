import serial
from crccheck.crc import Crc8, Crc16

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


    