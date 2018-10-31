import subprocess, struct, time, select, threading, os, sys, traceback, itertools,math, collections
import pytcpdump
import re
import tldextract

pcap_file = ['HTTPS-Identification-Framework/pcaps/GCDay1.pcap']
output_file_stats = 'ML_dataset/csvs/GCDay1stats.csv'
output_file_seqs = 'ML_dataset/csvs/GCDay1seq.csv'

def stat_head():
	return "sni,size25,size50,size75,sizeMax,sizeAvg,sizeVar,iat25,iat50,iat75,iatMax,iatAvg,iatVar,Csize25,Csize50,Csize75,CsizeMax,CsizeAvg,CsizeVar,Ciat25,Ciat50,Ciat75,CiatMax,CiatAvg,CiatVar\n"

def sequence_head(n):
	return "sni," + ','.join([str(i) for i in range(1,n)]) + "\n"


#TODO: replace these with numpy functions
def stat_calc(x, limit):
	try:
		l=min(len(x),limit)
		if l==0:
			return [str(a) for a in [0,0,0,0,0,0]]
		if l==1:
			return [str(a) for a in [x[0], x[0], x[0], x[0], x[0], 0]]
		if len(x)>limit:
			x=x[:limit]
			l=len(x)
		x = sorted(x)
		avg = float(sum(x)) / l
		var = sum([(xi - avg) ** 2 for xi in x]) / l
		return [str(a) for a in [x[int(round((l-1)/4.0))],x[int(round((l-1)/2.0))],x[int(round((l-1)*3/4.0))],max(x),avg,var] ]
	except Exception as e:
		traceback.print_exc()

def combine_at(sec, usec):
	l = len(sec)
	return [sec[i]+usec[i]*1e-6 for i in range(l)]

def stat_prepare_iat(sec,  usec, t=None):
	l = len(sec)
	if not t:	
		t = combine_at(sec, usec)
	return [t[i+1]-t[i] for i in range(l-1)]

def first_n_packet_within_time(arrival,first_n_second):
	limit = arrival[0]+first_n_second
	try:
		return next(i for i, x in enumerate(arrival) if x > limit)
	except:
		return len(arrival)

#TODO: include packet AND payload size stats as features?
def stat_create_first_n_second(data,filename, first_n_second):
	with open(filename,'wb') as f:
		f.write(stat_head())
		for id in data:
			item=data[id]

			sni=SNIModificationbyone(item[0])

			if sni == 'unknown' or sni == 'unknown.':
				continue

			line=[sni]
			#remote->local
			arrival = combine_at(item[2][0],item[3][0])
			first_n_packet = first_n_packet_within_time(arrival,first_n_second)
			line+=stat_calc(item[5][0], first_n_packet)
			iat = stat_prepare_iat(item[2][0],item[3][0], arrival)
			line+=stat_calc(iat, first_n_packet-1)
			
			#local->remote
			arrival = combine_at(item[2][1],item[3][1])
			first_n_packet = first_n_packet_within_time(arrival,first_n_second)
			line+=stat_calc(item[5][1], first_n_packet)
			iat = stat_prepare_iat(item[2][1],item[3][1], arrival)
			line+=stat_calc(iat, first_n_packet-1)
			line= ','.join(line)
			f.write(line)
			f.write('\n')

def sequence_create_first_n_packets(data, filename, first_n_packets):
	with open(filename,'wb') as f:
		f.write(sequence_head(first_n_packets))
		for id in data:
			item=data[id]

			sni=SNIModificationbyone(item[0])

			if sni == 'unknown' or sni == 'unknown.':
				continue

			#TODO: remove leading zeros in this case?
			if len(item[2][0]) + len(item[2][1]) > first_n_packets:
				print ("Too Many Packets: ", len(item[2][0]) + len(item[2][1]))

			line=[sni]

			#TODO: Also sort by milliseconds using combine_at function
			SCseq = zip(item[2][0], item[5][0])
			CSseq = zip(item[2][1], item[5][1])
			seq = [str(x) for _,x in sorted(SCseq + CSseq)]

			# Here is zero padding
			if len(seq) < first_n_packets:
				seq = [str(0)]*(first_n_packets - len(seq)) + seq

			line+=seq[0:first_n_packets]
			line= ','.join(line)
			f.write(line)
			f.write('\n')

def SNIModificationbyone(sni):
    temp = tldextract.extract(sni)
    x = re.sub("\d+", "", temp.subdomain)  # remove numbers
    x = re.sub("[-,.]", "", x)  #remove dashes
    x = re.sub("[www.,]", "", x) #remove www
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
	stat_create_first_n_second(pytcpdump.lru.cache, output_file_stats, first_n_second=5)
	sequence_create_first_n_packets(pytcpdump.lru.cache, output_file_seqs, first_n_packets=100)
