import cv2 as cv,js,numpy as np,scipy.fftpack as fftpack
from scipy.signal import detrend
def temporal_ideal_filter(arr,low,high,fps,axis=0):A=fftpack.fft(arr,axis=axis);B=fftpack.fftfreq(arr.shape[0],d=1.0/fps);C=np.abs(B-low).argmin();D=np.abs(B-high).argmin();A[:C]=0;A[D:-D]=0;A[-C:]=0;E=fftpack.ifft(A,axis=axis);return np.abs(E)
def reconstruct_video_g(amp_video,original_video,levels=3):
	D=original_video;C=amp_video;E=np.zeros(D.shape)
	for B in range(0,C.shape[0]):
		A=C[B]
		for F in range(levels):A=cv.pyrUp(A)
		A=A+D[B];E[B]=A
	return E
def build_gaussian_pyramid(src,levels=3):
	A=src.copy();B=[A]
	for C in range(levels):A=cv.pyrDown(A);B.append(A)
	return B
def gaussian_video(video,levels=3):
	B=video;C=B.shape[0]
	for A in range(0,C):
		F=build_gaussian_pyramid(B[A],levels=levels);D=F[-1]
		if A==0:E=np.zeros((C,*D.shape))
		E[A]=D
	return E
def find_heart_rate(vid,times,low,high,levels=3,alpha=20):
	A=times;B=vid.shape[0];F=B/(A[-1]-A[0]);J=magnify_color(vid,F,low,high,levels,alpha);K=np.mean(J,axis=(1,2,3));L=np.linspace(A[0],A[-1],B);M=detrend(K);C=np.interp(L,A,M);C=np.hamming(B)*C;N=C/np.linalg.norm(C);O=np.fft.rfft(N*30);D=float(F)/B*np.arange(B/2+1);E=60.0*D;G=np.abs(O)**2;H=np.where((E>50)&(E<180));I=G[H];P=E[H];D=P;G=I
	try:Q=np.argmax(I);R=D[Q];return R
	except ValueError:return 0
def magnify_color(vid,fps,low,high,levels=3,alpha=20):A=levels;B=gaussian_video(vid,levels=A);C=temporal_ideal_filter(B,low,high,fps);D=alpha*C;return reconstruct_video_g(D,vid,levels=A)
def canvas2numpy(canvas,context):A=canvas;B=context.getImageData(0,0,A.width,A.height);C=np.frombuffer(B.data.to_py(),np.uint8,-1);return C.reshape(A.height,A.width,4)[...,:3]
class HeartRateFinder:
	def __init__(A,num_frames,frame_width,frame_height):A.buffer_shape=num_frames,frame_height,frame_width,3;A.canvas=js.document.querySelector('canvas');A.ctx=A.canvas.getContext('2d');A.buffers=[];A.times=[];A.bpm=[];A.frame=None
	def collect_frame(A):A.frame=canvas2numpy(A.canvas,A.ctx)
	def get_heart_rate(A,face_id,box,timestamp):
		C=box;B=face_id
		if B>=len(A.buffers):A.buffers.append(np.zeros(A.buffer_shape));A.times.append(np.zeros((A.buffer_shape[0],)));A.bpm.append(np.zeros((A.buffer_shape[0],)))
		H,D,E,H=A.buffer_shape;F=int(C['x']+C['width']/2);G=int(C['y']+C['height']/2);A.buffers[B][:-1]=A.buffers[B][1:];A.buffers[B][-1]=A.frame[G-D//2:G+D//2,F-E//2:F+E//2];A.times[B][:-1]=A.times[B][1:];A.times[B][-1]=timestamp;A.bpm[B][:-1]=A.bpm[B][1:];A.bpm[B][-1]=find_heart_rate(vid=A.buffers[B],times=A.times[B],low=0.8333,high=1.0);return np.mean(A.bpm[B])