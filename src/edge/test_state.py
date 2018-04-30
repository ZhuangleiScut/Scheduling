# -*-coding:UTF-8 -*-
# get memory state
from numpy import long


def memory_stat():
    mem = {}
    f = open("/proc/meminfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) < 2: continue
        name = line.split(':')[0]
        var = line.split(':')[1].split()[0]
        mem[name] = long(var) * 1024.0
    mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached']
    return mem


# get CPU state
def cpu_stat():
    cpu = []
    cpuinfo = {}
    f = open("/proc/cpuinfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if line == '\n':
            cpu.append(cpuinfo)
            cpuinfo = {}
        if len(line) < 2: continue
        name = line.split(':')[0].rstrip()
        var = line.split(':')[1]
        cpuinfo[name] = var
    return cpu


# get load state
def load_stat():
    loadavg = {}
    f = open("/proc/loadavg")
    con = f.read().split()
    f.close()
    loadavg['lavg_1'] = con[0]
    loadavg['lavg_5'] = con[1]
    loadavg['lavg_15'] = con[2]
    loadavg['nr'] = con[3]
    loadavg['last_pid'] = con[4]
    return loadavg


# get Uptime
def uptime_stat():
    uptime = {}
    f = open("/proc/uptime")
    con = f.read().split()
    f.close()
    all_sec = float(con[0])
    MINUTE, HOUR, DAY = 60, 3600, 86400
    uptime['day'] = int(all_sec / DAY)
    uptime['hour'] = int((all_sec % DAY) / HOUR)
    uptime['minute'] = int((all_sec % HOUR) / MINUTE)
    uptime['second'] = int(all_sec % MINUTE)
    uptime['Free rate'] = float(con[1]) / float(con[0])
    return uptime


# get net state
def net_stat():
    net = []
    f = open("/proc/net/dev")
    lines = f.readlines()
    f.close()
    for line in lines[2:]:
        con = line.split()
        """ 
        intf = {} 
        intf['interface'] = con[0].lstrip(":") 
        intf['ReceiveBytes'] = int(con[1]) 
        intf['ReceivePackets'] = int(con[2]) 
        intf['ReceiveErrs'] = int(con[3]) 
        intf['ReceiveDrop'] = int(con[4]) 
        intf['ReceiveFifo'] = int(con[5]) 
        intf['ReceiveFrames'] = int(con[6]) 
        intf['ReceiveCompressed'] = int(con[7]) 
        intf['ReceiveMulticast'] = int(con[8]) 
        intf['TransmitBytes'] = int(con[9]) 
        intf['TransmitPackets'] = int(con[10]) 
        intf['TransmitErrs'] = int(con[11]) 
        intf['TransmitDrop'] = int(con[12]) 
        intf['TransmitFifo'] = int(con[13]) 
        intf['TransmitFrames'] = int(con[14]) 
        intf['TransmitCompressed'] = int(con[15]) 
        intf['TransmitMulticast'] = int(con[16]) 
        """
        intf = dict(
            zip(
                ('interface', 'ReceiveBytes', 'ReceivePackets',
                 'ReceiveErrs', 'ReceiveDrop', 'ReceiveFifo',
                 'ReceiveFrames', 'ReceiveCompressed', 'ReceiveMulticast',
                 'TransmitBytes', 'TransmitPackets', 'TransmitErrs',
                 'TransmitDrop', 'TransmitFifo', 'TransmitFrames',
                 'TransmitCompressed', 'TransmitMulticast'),
                (con[0].rstrip(":"), int(con[1]), int(con[2]),
                 int(con[3]), int(con[4]), int(con[5]),
                 int(con[6]), int(con[7]), int(con[8]),
                 int(con[9]), int(con[10]), int(con[11]),
                 int(con[12]), int(con[13]), int(con[14]),
                 int(con[15]), int(con[16]),)
            )
        )

        net.append(intf)
    return net


# get disk state
def disk_stat():
    import os
    hd = {}
    disk = os.statvfs("/")
    hd['available'] = disk.f_bsize * disk.f_bavail
    hd['capacity'] = disk.f_bsize * disk.f_blocks
    hd['used'] = disk.f_bsize * disk.f_bfree
    return hd


if __name__ == '__main__':
    memory = memory_stat()
    cpu = cpu_stat()
    load = load_stat()
    uptime = uptime_stat()
    net_state = net_stat()
    disk_state = disk_stat()
    print('memory:', memory)
    print('cpu:', cpu)
    print('load:', load)
    print('uptime:', uptime)
    print('net:', net_state)
    print('disk:', disk_state)
