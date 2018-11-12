#***********************************************************************************
# replicating what tcpdump terminal command does in python using appropriate filters
# For instance filters for https handshake packets and based on handshake packets gets the sni
#***********************************************************************************



import subprocess, struct, time, select, threading, os, sys, traceback, itertools, math, collections
from ctypes import *
from pytcpdump_utils import *

cache = None

#***********************************************************************************
# Pcap file header
#***********************************************************************************
class pcap_hdr_s(Structure):
	_fields_ = [('magic', c_uint32),
				('v1', c_uint16),
				('v2', c_uint16),
				('zone', c_uint32),
				('sigfigs', c_uint32),
				('snaplen', c_uint32),
				('network', c_uint32)]

#***********************************************************************************
# Packet headers
#***********************************************************************************
class pcaprec_hdr_s(Structure):
	_fields_ = [('sec', c_uint32),
				('usec', c_uint32),
				('len', c_uint32),
				('olen', c_uint32)]

#***********************************************************************************
# Cache stores data for each TCP handshake. 
# 
# Key:
# - connections ID
# 
# Values:
# - TCP data dictionary
#
# 0:str:hostname, 
# 1:[int,int]:accumulated bytes[remote->local, local->remote], 
# 2:list[[int],[int]]:arrival seconds, 
# 3:list[[int],[int]]:arrival micro-seconds, 
# 4:list[[int],[int]]:packet size
# 5:list[[int],[int]]:tcp payload size
#***********************************************************************************
class Cache:
	def __init__(self):
		self.lock = threading.RLock()
		self.cache = collections.OrderedDict()
	
	def set_hostname(self, key, hostname):
		with self.lock:
			item = self.pop(key)
			item[0] = hostname
			self.cache[key] = item
			
	def update(self, key, fromLocal, pcaprec_hdr, payload_len):
		with self.lock:
			item = self.pop(key)
			item[1][fromLocal]+=pcaprec_hdr.olen
			item[2][fromLocal].append(pcaprec_hdr.sec)
			item[3][fromLocal].append(pcaprec_hdr.usec)
			item[4][fromLocal].append(pcaprec_hdr.olen)
			item[5][fromLocal].append(payload_len)
			self.cache[key] = item
			
	def pop(self, key):
		with self.lock:
			try:
				return self.cache.pop(key)
			except KeyError:
				return ['unknown',[0,0],[[],[]],[[],[]],[[],[]],[[],[]]]

#***********************************************************************************
# Search for the SNI in the encrypted payload!
#***********************************************************************************
def sni_pos(data):
	pos=0
	while 1:
		pos=data.find(b'.',pos)
		if pos<0:
			return -1,-1
		
		# get start of domain
		pos1=pos
		while isDomainChar(data[pos1-1:pos1]):
			pos1-=1

		# get end of domain
		pos2=pos
		while 1:
			pos2+=1
			if pos2==len(data):
				break
			if not isDomainChar(data[pos2:pos2+1]):
				break

		# min domain length
		if pos2-pos1>=5:
			if ord(data[pos1-1:pos1])==0: #data[pos1-2:pos1] should be the len
				pos1+=1
			if (ord(data[pos1-2:pos1-1])<<8)+ord(data[pos1-1:pos1]) == pos2-pos1:
				return pos1, pos2
		
		# try again
		pos=pos2

#***********************************************************************************
# The meat and potatoes: 
# 1. Iterate through the pcap file
# 2. Get connection id
# 3. Add packet to connection id cache
# 4. Search for hostname in encrypted payload
#***********************************************************************************
def process_file(filename):
	global cache
	if not cache:
		cache = Cache()
	pcap_hdr = pcap_hdr_s()
	pkt_hdr = pcaprec_hdr_s()
	pkt_hdr_size = sizeof(pkt_hdr)
	with open(filename,'rb') as f:
		f.readinto(pcap_hdr)
		counter = 0
		while 1:
			counter = counter + 1
			if f.readinto(pkt_hdr)!=pkt_hdr_size:
				break

			# get next packet
			data =  bytes(f.read(pkt_hdr.len))
			ip_pos = get_ip_pos(data) # IP layer
			
			# Invalid IP shouldn't happen, but just in case...
			if ip_pos<0:
				print("Invalid IP: ", ip_pos)
				continue
			
			tcp_pos = ip_pos + ((ord(data[ip_pos:ip_pos+1]) & 0x0f)<<2)
			ip_proto = ord(data[ip_pos+9:ip_pos+10])
			
			# Must be TCP!
			if ip_proto != 0x06: 
				print("Skipping unexpected connection: ", ip_proto)
				continue
			
			conn_id = data[ip_pos+ 12: ip_pos+20] + data[tcp_pos : tcp_pos+4]
			conn_id, fromLocal = unify_conn_id(conn_id)
			tcp_load_pos = tcp_pos + (( ord(data[tcp_pos+12:tcp_pos+13]) & 0xf0 )>>2)
			cache.update(conn_id, fromLocal, pkt_hdr, pkt_hdr.olen-tcp_load_pos)
			if (tcp_load_pos+5<len(data)) and (ord(data[tcp_load_pos:tcp_load_pos+1]) == 0x16) and (ord(data[tcp_load_pos+5:tcp_load_pos+6]) == 0x01):
				pos1, pos2 = sni_pos(data)
				if pos1>0:
					cache.set_hostname(conn_id, data[pos1:pos2].decode())
