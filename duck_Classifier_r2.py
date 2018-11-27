# -*-encoding:utf-8-*-
'''
此版本加入
1.裝飾器功能，計算函數執行時間 timer()
2.修改判別函式以簡單結構化的貝葉斯判別分類
'''
import cv2
import numpy as np
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import math, os, random
import time
import threading, queue
from multiprocessing import Process, Queue

q_noduck_xy = queue.Queue()
read_dock = queue.Queue()


def timer(func):
	def wrapper(*args, **kwargs):
		print('%s 函數開始執行時間   :%s' % (func, time.asctime(time.localtime(time.time()))))
		start_time = time.time()
		func(*args, **kwargs)
		stop_time = time.time()
		print("%s 函數執行執行花費時間:%s" % (func, stop_time - start_time))
	
	return wrapper


def estimate_gaussian(dataset):
	''' 計算 MLE mu及sigma np公式計算'''
	mu = np.mean(dataset, axis=0)
	sigma = np.cov(dataset.T)
	return mu, sigma


def gaussian_mle(data):
	''' 計算 MLE mu及sigma  以高斯公式計算'''
	mu = data.mean(axis=0)
	var = (data - mu).T @ (data - mu) / (data.shape[0] - 1)  # this is slightly suboptimal, but instructive
	return mu, var


def multivariate_gaussian(dataset, mu, sigma):
	'''
	Caculate the multivariate normal density
	'''
	# p = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
	p = multivariate_normal(mean=mu, cov=sigma)
	return p.pdf(dataset)


def pdf_multivariate_gauss(x, mu, cov):
	'''
	Caculate the multivariate normal density (pdf)
	Keyword arguments:
		x   = numpy array of a "d x 1" sample vector
		mu  = numpy array of a "d x 1" mean vector
		cov = numpy array of a "d x d" covariance matrix
	'''
	
	part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
	part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
	return float(part1 * np.exp(part2))


def importdata(sample_dir):
	'''
	導入訓練集
	'''
	import glob
	p_dataset = []
	n_dataset = []
	
	for files in glob.glob(r".\data\p_" + sample_dir + "\*.jpg"):
		(B, G, R) = cv2.split(cv2.imread(files))  # 提取R、G、B分量
		p_dataset.append([B, G, R])
	
	for files in glob.glob(r".\data\n_" + sample_dir + "\*.jpg"):
		(B, G, R) = cv2.split(cv2.imread(files))  # 提取R、G、B分量
		n_dataset.append([B, G, R])
	
	return p_dataset, n_dataset


def train(dataset):
	'''
		從訓練集分別獲取不同類別下的平均向量、方差相關系數
	'''
	from sklearn.utils import shuffle
	
	s = dataset.shape[3]
	bmu_hat = np.ones(s)
	gmu_hat = np.ones(s)
	rmu_hat = np.ones(s)
	
	s = (dataset.shape[3], dataset.shape[3])
	mu_hat = []
	sigma_hat = []
	bsigma_hat = np.ones(s)
	gsigma_hat = np.ones(s)
	rsigma_hat = np.ones(s)
	
	for i in range(len(dataset)):
		bmu, bsig = gaussian_mle(dataset[i][0])
		gmu, gsig = gaussian_mle(dataset[i][1])
		rmu, rsig = gaussian_mle(dataset[i][2])
		
		bmu_hat += bmu
		gmu_hat += gmu
		rmu_hat += rmu
		bsigma_hat += bsig
		gsigma_hat += gsig
		rsigma_hat += rsig
	
	mu_hat.append(bmu_hat / len(dataset))
	mu_hat.append(gmu_hat / len(dataset))
	mu_hat.append(rmu_hat / len(dataset))
	
	sigma_hat.append(bsigma_hat / len(dataset))
	sigma_hat.append(gsigma_hat / len(dataset))
	sigma_hat.append(rsigma_hat / len(dataset))
	
	return mu_hat, sigma_hat


def split_data(data, prob):
	''' 將資料依prob數值 隨機分成 tran data 及 test data 	'''
	results = [], []
	for row in data:
		results[0 if random.random() < prob else 1].append(row)
	return results


def GaussianFunc1(dataset, mu, sigma):
	'''    根據指定參數mu、sigma計算多元變量的高斯函数    '''
	
	res = 0.0
	b = g = r = 0.0
	# 計算 B G R 鴨體的多變量高斯函數 帶入鴨體的平均變量及變異數
	for j in range(len(dataset[0])):
		X = dataset[0][j]
		b += pdf_multivariate_gauss(X, mu[0], sigma[0])
		X = dataset[1][j]
		g += pdf_multivariate_gauss(X, mu[1], sigma[1])
		X = dataset[2][j]
		r += pdf_multivariate_gauss(X, mu[2], sigma[2])
	res = b * g * r
	
	return res


def classify(dataset, pmu, nmu, psigma, nsigma):
	'''    基於貝葉斯判別分類　 '''
	res1 = GaussianFunc1(dataset, pmu, psigma)  # 傳入正樣本的mu、sigma
	res2 = GaussianFunc1(dataset, nmu, nsigma)  # 傳入負樣本的mu、sigma
	if res1 > res2:
		return 1
	else:
		return 0


def test(dataset, pmu, nmu, psigma, nsigma, pt=9):
	'''    測試    '''
	positive0 = 0
	positive1 = 0
	len_data = len(dataset)
	
	for item in dataset:
		res = classify(item, pmu, nmu, psigma, nsigma)
		if res == 0:
			positive0 += 1
		if res == 1:
			positive1 += 1
	if pt == 9:  # 回傳機率值
		print('bad_duck ', positive0 * 1.0 / len_data)
		print('good_duck', positive1 * 1.0 / len_data)
	if pt == 1:  # 回傳鴨子
		return positive1
	if pt == 0:  # 回傳非鴨子
		return positive0


@timer
def read_full_duck(img, yp_min, yp_max, xp_max, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat):
	''' 依據傳入的圖片Y軸的大小區間，去擷取圖片W x H大小
		判斷圖片是否為鴨體，若非則寫入Queue中去修改
	'''
	# 裁切區域的長度與寬度
	w = w
	h = h
	# 裁切圖片
	for j in range(yp_min, yp_max, h):
		x = 0
		y = j
		print(j)
		for i in range(0, xp_max, w):
			p_data = []
			x = i
			if x + w > xp_max: break
			crop_img = img[y:y + h, x:x + w]
			(B, G, R) = cv2.split(crop_img)  # 提取R、G、B分量
			p_data.append((B, G, R))
			p_data = np.array(p_data)
			dock = test(p_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat, 1)
			if dock == 0:  # 非鴨體
				img[y:y + h, x:x + w] = (0, 0, 0)  # 將非鴨體部分塗黑


@timer
def read_full_duck_thread(img, yp_min, yp_max, xp_max, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat):
	''' 依據傳入的圖片Y軸的大小區間，去擷取圖片W x H大小
		判斷圖片是否為鴨體，若非則寫入Queue中去修改
	'''
	read_dock.put(9)  # 寫入一筆執行記錄
	# 裁切區域的長度與寬度
	w = w
	h = h
	# 裁切圖片
	for j in range(yp_min, yp_max, h):
		x = 0
		y = j
		print(j)
		for i in range(0, xp_max, w):
			p_data = []
			x = i
			if x + w > xp_max: break
			crop_img = img[y:y + h, x:x + w]
			(B, G, R) = cv2.split(crop_img)  # 提取R、G、B分量
			p_data.append((B, G, R))
			p_data = np.array(p_data)
			dock = test(p_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat, 1)
			if dock == 0:  # 非鴨體
				q_noduck_xy.put((x, y))  # 非鴨體座標寫入Queue中，drow_full_duck讀取並執行
	
	a = read_dock.get()  # 刪除執行記錄


def drow_full_duck(duckimg, w, h):
	''' 將非鴨體的像素改成黑色像素'''
	while True:
		x, y = q_noduck_xy.get()
		w = w
		h = h
		duckimg[y:y + h, x:x + w] = (0, 0, 0)
		if q_noduck_xy.empty() and read_dock.empty():  # 無執行read_dock的執行緒，就中斷離開
			break


@timer
def read_full_duck_all(img, yp_min, yp_max, xp_max, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat):
	''' 依據傳入的圖片Y軸的大小區間，去擷取圖片WXH大小
		判斷圖片是否為鴨體，若非則寫入Queue中去修改
	'''
	# 裁切區域的長度與寬度
	w = w
	h = h
	x = 0
	y = yp_min
	# 裁切圖片
	while y + h < yp_max:
		p_data = []
		crop_img = img[y:y + h, x:x + w]
		(B, G, R) = cv2.split(crop_img)  # 提取R、G、B分量
		p_data.append((B, G, R))
		p_data = np.array(p_data)
		dock = test(p_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat, 1)
		if dock == 0:  # 非鴨體
			img[y:y + h, x:x + w] = (0, 0, 0)  # 將非鴨體部分塗黑
		
		x = x + w
		if x + w >= xp_max:
			x = 0
			y = y + h
			print(y)


if __name__ == '__main__':
	p_dataset, n_dataset = importdata('output8')  # 載入8x8正負樣本資料路徑
	# p_dataset, n_dataset = importdata('output4')  # 載入4x4正負樣本資料路徑
	# p_dataset, n_dataset = importdata('output2')  # 載入2x2正負樣本資料路徑
	
	p_dataset = np.array(p_dataset)
	n_dataset = np.array(n_dataset)
	
	random.seed(0)  # 偽亂數，可將split_data設成相同
	p_train_data, p_test_data = split_data(p_dataset, 0.7)  # 將正樣本資料分成tran及test
	n_train_data, n_test_data = split_data(n_dataset, 0.7)  # 將負樣本資料分成tran及test
	p_train_data = np.array(p_train_data)
	n_train_data = np.array(n_train_data)
	p_test_data = np.array((p_test_data))
	n_test_data = np.array((n_test_data))
	
	p_mu_hat, p_sigma_hat = train(p_train_data)  # 計算正樣本的mu及sigma
	n_mu_hat, n_sigma_hat = train(n_train_data)  # 計算負樣本的mu及sigma
	
	# 計算正樣本識別率
	# duck_prob = duck_test(p_test_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat)
	test(p_test_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat)
	
	# 計算負樣本識別率
	# nonduck_prob = duck_test(n_test_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat)
	test(n_test_data, p_mu_hat, n_mu_hat, p_sigma_hat, n_sigma_hat)
	
	# 讀入養鴨場全圖
	img = cv2.imread(r".\full_duck.jpg")
	
	# 圖檔的長度與寬度
	yp_length = img.shape[0]
	xp_length = img.shape[1]
	# 裁切區域的長度與寬度
	w = p_dataset.shape[2]
	h = p_dataset.shape[3]
	
	# 將之前樣本資料清空
	del p_dataset, n_dataset
	del p_test_data, n_test_data
	del p_train_data, n_train_data
	
	
	# 記錄開始時間，併入裝飾器中計算
	# 讀取養鴨場圖檔資料並將非鴨體設成黑色 while loop
	read_full_duck_all(img, 0, yp_length, xp_length, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat)
	# 讀取養鴨場圖檔資料並將非鴨體設成黑色 for loop
	#read_full_duck(img, 0, yp_length, xp_length, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat)
	
	# 將結果存成新圖檔
	cv2.imwrite(r".\Result_duck.jpg", img)
	
	
	# 以下是以執行緒方式
	#
	# img = cv2.imread(r".\full_duck.jpg")
	#
	# # 建立數個執行緒去執行判別替換所有非鴨體像素為黑色像素
	# r1 = threading.Thread(target=read_full_duck_thread, args=(img, 0, 4000, xp_length, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat))
	# r2 = threading.Thread(target=read_full_duck_thread, args=(img, 4000, 8000, xp_length, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat))
	# r3 = threading.Thread(target=read_full_duck_thread, args=(img, 8000, yp_length, xp_length, w, h, p_mu_hat, p_sigma_hat, n_mu_hat, n_sigma_hat))
	# r7 = threading.Thread(target=drow_full_duck, args=(img, w, h))
	#
	# r1.start()
	# r2.start()
	# r3.start()
	# r7.start()
	#
	# r1.join()
	# r2.join()
	# r3.join()
	# r7.join()
	#
	# # 輸出新的圖像
	# cv2.imwrite(r".\duckCopy2.jpg", img1)
	#