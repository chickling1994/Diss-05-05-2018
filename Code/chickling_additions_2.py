#!/usr/bin/env python3
import os, os.path
import sys
import time
import csv

import numpy as np
import scipy.signal as sig

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.axes as ax
import scipy

from scipy.fftpack import fft
from scipy import signal

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def file_finder(shotno, date=time.strftime("%Y%m%d")):
	# try to find the file for the shotnumber requested
	# a date can be entered in the format chickling_eval(shotno, YYYYMMDD) and shots from that day will be found. If no date is entered todays date will be the default.
	fname = None
	for name in os.listdir(D_FOLDER) :
		if str(date)+ "_SHOTNO_" + str(shotno)+".dat" in name :
			fname = D_FOLDER + '/' + name
			break
	
	print(fname)
	
	#Jakob's function to extract the data from the .dat file.
	data = quickextract_data(fname)
	
	return fname, data

def chickling_eval(shotno, date=time.strftime("%Y%m%d")):

	fname, data = file_finder(shotno,date)
	
	fs = data[0]['samplerate']

	plt.figure("Phase Difference CO2 shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(data[0]['timetag_phdiff']/fs,np.unwrap(data[0]['phasediff_co2']))
	

	plt.figure("Scene Signal CO2 shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Amplitude, V")
	plt.plot(data[0]['timetag_mags']/fs,data[0]['co2_sig_mag'])

	plt.figure("Ref Signal CO2 shot " + str(shotno) +   " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Amplitude, V")
	plt.plot(data[0]['timetag_mags']/fs,data[0]['co2_ref_mag'])

def chickling_pd(shotno, date=time.strftime("%Y%m%d")):
	fname, data = file_finder(shotno,date)

	fs = data[0]['samplerate']
	plt.figure("Phase Difference shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(data[0]['timetag_phdiff'][1:]/fs,np.trim_zeros(np.unwrap(data[0]['phasediff_co2'][1:])))
	print(np.std(np.unwrap(data[0]['phasediff_co2'])))
	
def chickling_pd_zoom(shotno, date=time.strftime("%Y%m%d")):
	
	fname, data = file_finder(shotno,date)
	
	data_1550 = data[0]['phasediff_co2'][100:]
	plot_time = np.linspace(0,1,data_1550.size)

	
	fig, ax = plt.subplots()
	ax.plot(plot_time, data_1550)
	ax.set_ybound(max(data_1550)+0.6, min(data_1550)-0.01)
	
	plt.title("Phase Difference for shot " + str(shotno) + " Date " +  str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians") 	
	
	x_zoom_bot = int(data_1550.size*(10/100))
	x_zoom_top = int(data_1550.size*(15/100))
	

	x1, x2, y1, y2 = 0.1, 0.15, max(data_1550[x_zoom_bot:x_zoom_top])+0.01, min(data_1550[x_zoom_bot:x_zoom_top])-0.01
	
	axins = inset_axes(ax, 4.3,1, loc=9)
	axins.plot(plot_time[x_zoom_bot:x_zoom_top], data_1550[x_zoom_bot:x_zoom_top])

	axins.set_xlim(x1, x2)
	if y1 < y2:
		axins.set_ylim(y1, y2)
	else:
		axins.set_ylim(y2, y1)

	mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",lw=2)
	
	plt.show()

def chickling_pd_2ch(shotno, date=time.strftime("%Y%m%d")):

	fname, data = file_finder(shotno,date)
	
	fs = data[0]['samplerate']
	plt.figure("Phase Difference Scene shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(data[0]['timetag_phdiff']/fs,np.unwrap(data[0]['phasediff_co2']))
	plt.figure("Phase Difference Ref shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(data[0]['timetag_phdiff']/fs,np.unwrap(data[0]['phasediff_hene']))

def chickling_pd_overlay(shotno, date=time.strftime("%Y%m%d")):

	fname, data = file_finder(shotno,date)
	
	fs = data[0]['samplerate']
	plt.figure("Phase Difference Blue = Scene Orange = Reference shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(data[0]['timetag_phdiff']/fs,np.unwrap(data[0]['phasediff_co2']-np.average(np.unwrap(data[0]['phasediff_co2']))))
	plt.plot(data[0]['timetag_phdiff']/fs,np.unwrap(data[0]['phasediff_hene']-np.average(np.unwrap(data[0]['phasediff_hene']))))
	
def chickling_pn(shotno, date=time.strftime("%Y%m%d"), bandwidth=1000):
	    
	fname, data = file_finder(shotno,date)
	
	fs = data[0]['samplerate']	
	samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidth)
	phase_noise_sc = np.zeros(samplesize)
	phase_noise_ref = np.zeros(samplesize)

	#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
	phasediff_pn_sc = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))
	phasediff_pn_ref = np.reshape(np.unwrap(data[0]['phasediff_hene'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))

	#for each horizontal column perform a standard deviation
	for i in range(0,samplesize):
		phase_noise_sc[i] = np.std(phasediff_pn_sc[i])
		phase_noise_ref[i] = np.std(phasediff_pn_ref[i])

	#plot STD against time and find the average
	plt.figure("Phase Noise for Scene shot " + str(shotno) + " with bandwidth = " + str(bandwidth) +   " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Noise, mRadians")
	plt.plot(np.linspace(0,1,samplesize),phase_noise_sc*1000)
	print("Scene Phase STD = "+str(np.std(phase_noise_sc)))
	print("Scene Phase AVR = "+str(np.mean(phase_noise_sc)))

	plt.figure("Phase Noise for Ref shot " + str(shotno) + " with bandwidth = " + str(bandwidth) +   " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Noise, mRadians")
	plt.plot(np.linspace(0,1,samplesize),phase_noise_ref*1000)
	print("Ref Phase STD = "+str(np.std(phase_noise_ref)))
	print("Ref Phase AVR = "+str(np.mean(phase_noise_ref)))

def chickling_avr(shotno, date=time.strftime("%Y%m%d"), bandwidth=40000):
	    
	fname, data = file_finder(shotno,date)
	
	fs = data[0]['samplerate']	
	samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidth)
	phase_avr = np.zeros(samplesize)

	#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
	phasediff_avr = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))

	#for each horizontal column perform a standard deviation
	for i in range(1,samplesize):
		phase_avr[i] = np.mean(phasediff_avr[i])

	#plot STD against time and find the average
	plt.figure("Average Phase Difference for shot " + str(shotno) + " with bandwidth = " + str(bandwidth) +  " Date " +  str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, mRadians") 
	plt.plot(phase_avr)
	print("Phase STD = "+str(np.std(phase_avr)))
	print("Drift = "+str(np.mean(phase_avr[2:10])-np.mean(phase_avr[int(phase_avr.size-8):]) ) + " Radians")
	
def chickling_avr_2ch(shotno, date=time.strftime("%Y%m%d"), bandwidth=40000):
	    
	fname, data = file_finder(shotno,date)
	
	samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidth)
	phase_avr_co2 = np.zeros(samplesize)
	phase_avr_hene = np.zeros(samplesize)

	#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
	phasediff_co2 = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))
	phasediff_hene = np.reshape(np.unwrap(data[0]['phasediff_hene'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))

	#for each horizontal column perform an average
	for i in range(0,samplesize):
		phase_avr_co2[i] = np.mean(phasediff_co2[i])
		phase_avr_hene[i] = np.mean(phasediff_hene[i])
	
	x = np.linspace(0,1,samplesize)
	plt.figure("2 Channels | Blue = Scene | Orange = Reference | shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(x,phase_avr_co2-np.average(phase_avr_co2))
	plt.plot(x,phase_avr_hene-np.average(phase_avr_hene))
	plt.show()
	
def chickling_corr(shotno, date=time.strftime("%Y%m%d"), bandwidth=40000):

	fname, data = file_finder(shotno,date)
		
	samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidth)
	phase_avr_co2 = np.zeros(samplesize)
	phase_avr_hene = np.zeros(samplesize)

	#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
	phasediff_co2 = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))
	phasediff_hene = np.reshape(np.unwrap(data[0]['phasediff_hene'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))

	#for each horizontal column perform an average
	for i in range(0,samplesize):
		phase_avr_co2[i] = np.mean(phasediff_co2[i])
		phase_avr_hene[i] = np.mean(phasediff_hene[i])

	x = np.linspace(0,1,samplesize)
	plt.figure("2 Channels | Blue = Scene | Orange = Reference | Green = Cross-Correlation | shot " + str(shotno) +  " Date " + str(date))
	plt.xlabel("Time, s")
	plt.ylabel("Phase Difference, Radians")
	plt.plot(x,phase_avr_co2-np.average(phase_avr_co2))
	plt.plot(x,phase_avr_hene-np.average(phase_avr_hene))

	a = (phase_avr_co2 - np.mean(phase_avr_co2)) / (np.std(phase_avr_co2) * len(phase_avr_co2))
	b = (phase_avr_hene - np.mean(phase_avr_hene)) / (np.std(phase_avr_hene))
	yc = np.correlate(a, b, 'full')
	print(np.correlate(a, b, 'valid'))
	xc = np.linspace(0,1,yc.size)
	plt.plot(xc,yc)#,'o',ms=0.4)
	#plt.xcorr(a,b)#,'o',ms=0.4)

	#plt.figure("Correlations of Shot "+str(shotno)+" to "+ str(shotno+fileno))
	#plt.plot(np.linspace(0,1,fileno),correlation,'o')

def chickling_corr_2ch(shotno,fileno, date=time.strftime("%Y%m%d"), bandwidth=40000):
	
	corr = np.zeros(fileno)
	for j in range (0,  fileno):
		try:
			fname, data = file_finder(shotno,date)
			samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidth)
			phase_avr_co2 = np.zeros(samplesize)
			phase_avr_hene = np.zeros(samplesize)

			#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
			phasediff_co2 = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))
			phasediff_hene = np.reshape(np.unwrap(data[0]['phasediff_hene'][0:(samplesize*bandwidth)]),(samplesize,bandwidth))

			#for each horizontal column perform an average
			for i in range(0,samplesize):
				phase_avr_co2[i] = np.mean(phasediff_co2[i])
				phase_avr_hene[i] = np.mean(phasediff_hene[i])

			a = (phase_avr_co2 - np.mean(phase_avr_co2)) / (np.std(phase_avr_co2) * len(phase_avr_co2))
			b = (phase_avr_hene - np.mean(phase_avr_hene)) / (np.std(phase_avr_hene))
			corr[j] = np.correlate(a, b, 'valid')


			#plt.xcorr(a,b)#,'o',ms=0.4)

			#plt.figure("Correlations of Shot "+str(shotno)+" to "+ str(shotno+fileno))
			#plt.plot(np.linspace(0,1,fileno),correlation,'o')
		except Exception:
			print("~~~~~ Encountered Error, Skipping Data Set ~~~~~")
			pass
	plt.figure("Correlation for shots "+str(shotno)+" to "+str(shotno+fileno))
	plt.plot(corr,'o')

def chickling_csd_2ch(shotno, date=time.strftime("%Y%m%d")):

	fname, data = file_finder(shotno,date)
	data = quickextract_data(fname)

	#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
	phasediff_co2 = np.unwrap(data[0]['phasediff_co2'])
	phasediff_hene = np.unwrap(data[0]['phasediff_hene'])
	fs = data[0]['samplerate']
	
	plt.figure("2 Channels | Blue = PSD Scene | Orange = PSD Reference | Green = CSD | shot " + str(shotno) +  " Date " + str(date))
	plt.psd(phasediff_co2, Fs=fs)
	plt.psd(phasediff_hene,Fs=fs)
	plt.csd(phasediff_co2, phasediff_hene, Fs=fs)
	plt.show()
	
def chickling_avr_pn(shotno, fileno=1, date=time.strftime("%Y%m%d"), bandwidthavr=40000,bandwidthstd=1000):    
	csv_drift = np.zeros(fileno)	
	csv_drift_std = np.zeros(fileno)
	csv_drift_mm = np.zeros(fileno)		
	csv_pn = np.zeros(fileno)	
	csv_pn_std = np.zeros(fileno)
	j_skipped = np.zeros(fileno)
	k=0
	for j in range (0,  fileno):
		try:
			fname, data = file_finder(shotno,date)
			fs = data[0]['samplerate']	
			samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/bandwidthavr)
			phase_avr = np.zeros(samplesize)
	
			#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
			phasediff_avr = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(samplesize*bandwidthavr)]),(samplesize,bandwidthavr))
	
			#for each horizontal column perform a standard deviation
			for i in range(1,samplesize):
				phase_avr[i] = np.mean(phasediff_avr[i])
	
			#plot STD against time and find the average
			#plt.figure("Average Phase Difference for shot " + str(shotno) + " with bandwidthavr = " + str(bandwidthavr) +  " Date " +  str(date))
			#plt.xlabel("Time, s")
			#plt.ylabel("Phase Difference, mRadians") 
			#plt.plot(phase_avr)
			print("Drift = "+str((np.mean(phase_avr[2:10])-np.mean(phase_avr[int(phase_avr.size-8):]) )*1000))		
			print("Drift STD = "+str(1000*np.std(phase_avr)))
		
	
			pn_bw = bandwidthstd
			pn_samplesize = int(np.unwrap(data[0]['phasediff_co2']).size/pn_bw)
			phase_noise = np.zeros(pn_samplesize)

			#reshape the array of x points (20M for 1s) into a 2d array each with 40k segments.
			phasediff_pn = np.reshape(np.unwrap(data[0]['phasediff_co2'][0:(pn_samplesize*pn_bw)]),(pn_samplesize,pn_bw))

			#for each horizontal column perform a standard deviation
			for i in range(1,pn_samplesize):
				phase_noise[i] = np.std(phasediff_pn[i])

			#plot STD against time and find the average
			#plt.figure("Phase Noise for Scene shot " + str(shotno) + " with bandwidth = " + str(pn_bw) +   " Date " + str(date))
			#plt.xlabel("Time, s")
			#plt.ylabel("Phase Noise, mRadians")
			#plt.plot(np.linspace(0,1,pn_samplesize),phase_noise*1000)
			print("Phase Noise STD = "+str(1000*np.std(phase_noise)))
			print("Phase Noise AVR = "+str(1000*np.mean(phase_noise)))

			csv_drift[j] = 	str((np.mean(phase_avr[2:10])-np.mean(phase_avr[int(phase_avr.size-8):]) )*1000)
			csv_drift_std[j] = str(1000*np.std(phase_avr))
			csv_drift_mm[j] = str(1000*abs(max(phase_avr[1:phase_avr.size])-min(phase_avr[1:phase_avr.size])))		
			csv_pn[j] = str(1000*np.mean(phase_noise))
			csv_pn_std[j] = str(1000*np.std(phase_noise))
			print("Max - Min Drift = " + str(csv_drift_mm[j]))
		except Exception:
			print("~~~~~ Encountered Error, Skipping Data Set ~~~~~")
			j_skipped[k] = j
			k=k+1
			pass
			

			
	with open(str(shotno)+'-'+str(shotno+fileno)+" at bandwidth "+str(bandwidthstd)+'.csv','w', newline='') as csvfile:
		for i in range(0, fileno):
			datawriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
			datawriter.writerow((csv_drift[i],csv_drift_std[i],csv_drift_mm[i],csv_pn[i],csv_pn_std[i]))
			#datawriter.writerow((bytes(csv_drift[i],"utf-8"),bytes(csv_drift_std[i],"utf-8"),bytes(csv_pn[i],"utf-8"),bytes(csv_pn_std[i],"utf-8")))
			#datawriter.writerow(b"hello")
	print("Number of Successful transfers: " + str(fileno - k - 1) +"/" +str(fileno))
	if max(j_skipped) > 0 :
		print("Data sets skipped are:")
		print(np.nonzero(j_skipped))
	print(csv_drift)
	print(csv_drift_std)
	print(csv_pn)
	print(csv_pn_std)
		
	
def chickling_csv(shotno, fileno=1, date=time.strftime("%Y%m%d")):
	for j in range (0, fileno):
	
		fname, data = file_finder(shotno,date)

		samplesize = int(np.unwrap(data[0]['phasediff_co2']).size)
		x_data = np.linspace(0,1,samplesize)
		y_data = np.unwrap(data[0]['phasediff_co2'])
		with open(str(shotno)+'-'+str(shotno+fileno)+date+'.csv','w', newline='') as csvfile:
			for i in range(0, samplesize):
				datawriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
				datawriter.writerow((x_data[i],y_data[i]))

def chickling_fft(shotno, date=time.strftime("%Y%m%d"),dec="FALSE"):
	
	fname, data = file_finder(shotno,date)
	if dec == "TRUE":
			phase_diff_fft = np.fft.rfft(sig.decimate(data[0]['phasediff_co2'],10))
	else:
		phase_diff_fft = np.fft.rfft(data[0]['phasediff_co2'])
	
	plt.figure("FFT for shot " + str(shotno) +   " Date " + str(date))
	plt.xlabel('frequency [Hz]')
	plt.ylabel("Amplitude, Arb. Units") 
	plt.loglog(np.abs(phase_diff_fft))
	plt.grid(b=True,which='both')

def chickling_csd(shotno, date=time.strftime("%Y%m%d")):
	    
	fname, data = file_finder(shotno,date)
	

	x = np.unwrap(data[0]['phasediff_co2'])
	y = np.unwrap(data[0]['phasediff_hene'])
	f, Pxy = signal.csd(x, y, 20e6, nperseg=1024)
	plt.figure("CSD for shot " + str(shotno) +   " Date " + str(date))
	plt.semilogy(f, np.abs(Pxy))
	plt.xlabel('frequency [Hz]')
	plt.ylabel('CSD [V**2/Hz]')
	#plt.grid(b=True,which='both')
