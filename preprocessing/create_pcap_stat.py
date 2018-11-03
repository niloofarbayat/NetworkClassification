import subprocess, struct, time, select, threading, os, sys, traceback, itertools,math, collections
import pytcpdump
import re
import tldextract
import numpy as np

pcap_file = ['../HTTPS-Identification-Framework/pcaps/GCDay1.pcap']
output_file_stats = '../ML/training/GCDay1stats.csv'
output_file_seqs = '../DL/training/GCDay1seq.csv'

def stat_head():
	return "sni,CSPktNum,CSPktsize25,CSPktSize50,CSPktSize75,CSPktSizeMax,CSPktSizeAvg,CSPktSizeVar,CSPaysize25,CSPaySize50,CSPaySize75,CSPaySizeMax,CSPaySizeAvg,CSPaySizeVar,CSiat25,CSiat50,CSiat75,SCPktNum,SCPktsize25,SCPktSize50,SCPktSize75,SCPktSizeMax,SCPktSizeAvg,SCPktSizeVar,SCPaysize25,SCPaySize50,SCPaySize75,SCPaySizeMax,SCPaySizeAvg,SCPaySizeVar,SCiat25,SCiat50,SCiat75,PktNum,Pktsize25,PktSize50,PktSize75,PktSizeMax,PktSizeAvg,PktSizeVar,Paysize25,PaySize50,PaySize75,PaySizeMax,PaySizeAvg,PaySizeVar,iat25,iat50,iat75\n"

def sequence_head(n):
	return "sni," + ','.join([str(i) for i in range(1,n)]) + "\n"

def stat_calc(x, iat=False):
	if len(x)==0:
		return [str(a) for a in [0,0,0,0,0,0]]
	if len(x)==1:
		return [str(a) for a in [x[0], x[0], x[0], x[0], x[0], 0]]
	x = sorted(x)
	p25,p50,p75 = get_percentiles(x)
	return [str(a) for a in [p25,p50,p75,max(x),np.mean(x),np.var(x)]]

def get_percentiles(x):
	return x[int(round((len(x)-1)/4.0))], x[int(round((len(x)-1)/2.0))], x[int(round((len(x)-1)*3/4.0))]

def combine_at(sec, usec):
	l = len(sec)
	return [sec[i]+usec[i]*1e-6 for i in range(l)]

def stat_prepare_iat(t):
	l = len(t)
	iat = [t[i+1]-t[i] for i in range(l-1)]
	if len(iat)==0:
		return [str(a) for a in [0,0,0]]
	if len(iat)==1:
		return [str(a) for a in [iat[0], iat[0], iat[0]]]
	p25,p50,p75 = get_percentiles(iat)
	return [str(a) for a in [p25,p50,p75]]

def stat_create(data,filename):
	with open(filename,'w') as f:
		f.write(stat_head())
		for id in data:
			item=data[id]

			sni=SNIModificationbyone(item[0])

			if sni == 'unknown' or sni == 'unknown.':
				continue

			line=[sni]
			#remote->local
			line+=[str(len(item[4][0]))]
			line+=stat_calc(item[4][0])
			line+=stat_calc(item[5][0])
			arrival1=combine_at(item[2][0], item[3][0])
			line+=stat_prepare_iat(arrival1)
			
			#local->remote
			line+=[str(len(item[4][1]))]
			line+=stat_calc(item[4][1])
			line+=stat_calc(item[5][1])
			arrival2=combine_at(item[2][1], item[3][1])
			line+=stat_prepare_iat(arrival2)

			line+=[str(len(item[4][1]) + len(item[4][0]))]
			line+=stat_calc(item[4][1] + item[4][0])
			line+=stat_calc(item[5][1] + item[5][0])
			line+=stat_prepare_iat(sorted(arrival1 + arrival2))

			line= ','.join(line)
			f.write(line)
			f.write('\n')

def sequence_create(data, filename, first_n_packets):
	with open(filename,'w') as f:
		f.write(sequence_head(first_n_packets))
		counter = 0
		for id in data:
			item=data[id]
			sni=SNIModificationbyone(item[0])

			if sni == 'unknown' or sni == 'unknown.':
				continue

			#TODO: remove leading zeros in this case?
			if len(item[2][0]) + len(item[2][1]) > first_n_packets:
				print("Too Many Packets: ", len(item[2][0]) + len(item[2][1]))

			line=[sni]

			arrival1=combine_at(item[2][0], item[3][0])
			arrival2=combine_at(item[2][1], item[3][1])
			
			seq = zip(arrival1 + arrival2, list(item[5][0]) + list(item[5][1]))
			seq = [str(x) for _,x in sorted(seq)]

			# Here is zero padding
			if len(seq) < first_n_packets:
				seq = [str(0)]*(first_n_packets - len(seq)) + seq

			line+=seq[0:first_n_packets]
			line= ','.join(line)
			f.write(line)
			f.write('\n')

#***********************************************************************************
#SNi modification for the sub-domain parts only
#***********************************************************************************
def SNIModificationbyone(sni):
    temp = tldextract.extract(sni.encode().decode())
    x = re.sub("\d+", "", temp.subdomain)  # remove numbers
    x = re.sub("[-,.]", "", x)  #remove dashes
    x = re.sub("[(?:www.)]", "", x) #remove www
    if len(x) > 0:
        newsni = x + "." + temp.domain + "." + temp.suffix  # reconstruct the sni
    else:
        newsni = temp.domain + "." + temp.suffix

    return newsni

for fname in pcap_file:
	print ('process', fname)
	pytcpdump.process_file(fname)
	print (fname,"finished, kept",len(pytcpdump.lru.cache),'records')

if __name__ == "__main__":
	stat_create(pytcpdump.lru.cache, output_file_stats)
	sequence_create(pytcpdump.lru.cache, output_file_seqs, first_n_packets=100)
