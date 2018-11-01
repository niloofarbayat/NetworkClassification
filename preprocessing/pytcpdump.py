import subprocess, struct, time, select, threading, os, sys, traceback, itertools,math, collections
from ctypes import *
from pytcpdump_utils import *

snaplen = '1600'
LRU_size = 100000 #TODO: This could be a problem
lru = None
local_ip = prob_local_ip()

class pcap_hdr_s(Structure):
	_fields_ = [('magic', c_uint32),
				('v1', c_uint16),
				('v2', c_uint16),
				('zone', c_uint32),
				('sigfigs', c_uint32),
				('snaplen', c_uint32),
				('network', c_uint32)]
class pcaprec_hdr_s(Structure):
	_fields_ = [('sec', c_uint32),
				('usec', c_uint32),
				('len', c_uint32),
				('olen', c_uint32)]


#index help:
#0:str:hostname, 
#1:[int,int]:accumulated bytes[remote->local, local->remote], 
#2:list[[int],[int]]:arrival seconds, 
#3:list[[int],[int]]:arrival micro-seconds, 
#4:list[[int],[int]]:packet size
#5:list[[int],[int]]:tcp payload size
class LRUCache:
	def __init__(self, capacity):
		self.capacity = capacity
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

	def get(self, key):
		with self.lock:
			try:
				value = self.cache.pop(key)
				self.cache[key] = value
				return value
			except KeyError:
				return -1

	def set(self, key, value):
		with self.lock:
			try:
				self.cache.pop(key)
			except KeyError:
				if len(self.cache) >= self.capacity:
					self.cache.popitem(last=False)
			self.cache[key] = value

def sni_pos(data):
	pos=0
	while 1:
		pos=data.find('.',pos)
		if pos<0:
			return -1,-1
		pos1=pos
		while isDomainChar(data[pos1-1]):
			pos1-=1
		pos2=pos
		while 1:
			pos2+=1
			if pos2==len(data):
				break
			if not isDomainChar(data[pos2]):
				break
		if pos2-pos1>=5:
			if ord(data[pos1-1])==0: #data[pos1-2:pos1] should be the len
				pos1+=1
			if (ord(data[pos1-2])<<8)+ord(data[pos1-1]) == pos2-pos1:
				#print (pkt_hdr.sec, pkt_hdr.usec, data[pos1:pos2], the_time)
				return pos1, pos2
		pos=pos2
	
def process_file(filename):
	global lru
	if not lru:
		lru = LRUCache(LRU_size)
	pcap_hdr = pcap_hdr_s()
	pkt_hdr = pcaprec_hdr_s()
	pkt_hdr_size = sizeof(pkt_hdr)
	with open(filename,'rb') as f:
		f.readinto(pcap_hdr)
		while 1:
			if f.readinto(pkt_hdr)!=pkt_hdr_size:
				break
			#print (pkt_hdr.len, pkt_hdr.olen, pkt_hdr.sec, pkt_hdr.usec)
			data = f.read(pkt_hdr.len)

			ip_pos = get_ip_pos(data) # IP layer
			if ip_pos<0:
				continue
			tcp_pos = ip_pos + ((ord(data[ip_pos]) & 0x0f)<<2) #mostly equals ip_pos+20 # TCP layer
			ip_proto = ord(data[ip_pos+9])
			if  ip_proto == 0x06: #tcp
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 't'
			elif ip_proto == 0x17: #udp
				continue
				conn_id = data[ip_pos+ 12: ip_pos+20]+data[tcp_pos : tcp_pos+4] + 'u'
			else:
				continue
			conn_id , fromLocal = unify_conn_id(conn_id, local_ip)
			tcp_load_pos = tcp_pos + (( ord(data[tcp_pos+12]) & 0xf0 )>>2)
			lru.update(conn_id, fromLocal, pkt_hdr, pkt_hdr.olen-tcp_load_pos)
			if (tcp_load_pos+5<len(data)) and ( ord(data[tcp_load_pos]) == 0x16) and (ord(data[tcp_load_pos+5]) == 0x01):
				#print ('https hello packet')
				pos1, pos2 = sni_pos(data)
				if pos1>0:
					lru.set_hostname(conn_id, data[pos1:pos2])
