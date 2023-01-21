import cv2 as cv
import js
import numpy as np
import scipy.fftpack as fftpack
from scipy.signal import detrend


def temporal_ideal_filter(arr, low, high, fps, axis=0):
    """ Applies a temporal ideal filter to a numpy array

    Paremeters
    ----------
    - arr: a numpy array with shape (N, H, W, C)
        - N: number of frames
        - H: height
        - W: width
        - C: channels
    - low: the low frequency bound
    - high: the high frequency bound
    - fps: the video frame rate
    - axis: the axis of video, should always be 0

    Returns
    -------
    - the array with the filter applied
    """
    fft = fftpack.fft(arr, axis=axis)
    frequencies = fftpack.fftfreq(arr.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def reconstruct_video_g(amp_video, original_video, levels=3):
    """ reconstructs a video from a gaussian pyramid and the original

    Parameters
    ----------
    - amp_video: the amplified gaussian video
    - original_video: the original video
    - levels: the levels in the gaussian video

    Returns
    --------
    - the reconstructed video
    """
    final_video = np.zeros(original_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img = cv.pyrUp(img)
        img = img + original_video[i]
        final_video[i] = img
    return final_video


def build_gaussian_pyramid(src, levels=3):
    """ Builds a gaussian pyramid

    Parameters
    ----------
    - src: the input image
    - levels: the number levels in the gaussian pyramid

    Returns
    -------
    - A gaussian pyramid
    """
    s = src.copy()
    pyramid = [s]
    for i in range(levels):
        s = cv.pyrDown(s)
        pyramid.append(s)
    return pyramid


def gaussian_video(video, levels=3):
    """ generates a gaussian pyramid for each frame in a video

    Parameters
    ----------
    - video: the input video array
    - levels: the number of levels in the gaussian pyramid

    Returns
    -------
    - the gaussian video
    """
    n = video.shape[0]
    for i in range(0, n):
        pyr = build_gaussian_pyramid(video[i], levels=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data = np.zeros((n, *gaussian_frame.shape))
        vid_data[i] = gaussian_frame
    return vid_data


def find_heart_rate(vid, times, low, high, levels=3, alpha=20):
    """ calculates the heart rate of a given face video

    Parameters
    ----------
    - vid: the video to find the heart rate in
    - times: the timestamps (in seconds) for each frame of the video
    - low: the lower wavelength to filter for
    - high: the higher wavelength to filter for
    - levels: the number of gaussian pyramid levels to compute
    - alpha: the amplification factor for magnifying a video

    Return
    ------
    - heart_rate
    """
    num_frames = vid.shape[0]
    true_fps = num_frames / (times[-1] - times[0])

    res = magnify_color(vid, true_fps, low, high, levels, alpha)

    avg = np.mean(res, axis=(1, 2, 3))
    even_times = np.linspace(times[0], times[-1], num_frames)

    processed = detrend(avg)#detrend the signal to avoid interference of light change
    interpolated = np.interp(even_times, times, processed) #interpolation by 1
    interpolated = np.hamming(num_frames) * interpolated#make the signal become more periodic (advoid spectral leakage)
    norm = interpolated/np.linalg.norm(interpolated)
    raw = np.fft.rfft(norm*30)

    freqs = float(true_fps) / num_frames * np.arange(num_frames / 2 + 1)
    freqs_ = 60. * freqs

    fft = np.abs(raw)**2#get amplitude spectrum

    idx = np.where((freqs_ > 50) & (freqs_ < 180))#the range of frequency that HR is supposed to be within
    pruned = fft[idx]
    pfreq = freqs_[idx]

    freqs = pfreq
    fft = pruned

    try:
        idx2 = np.argmax(pruned)#max in the range can be HR

        bpm = freqs[idx2]
        return bpm
    except ValueError:
        return 0


def magnify_color(vid, fps, low, high, levels=3, alpha=20):
    """ Magnifies the color of a video

    Parameters
    ----------
    - vid: the input video as a numpy array
    - fps: the frame rate of the video
    - low: the low frequency band to amplify
    - high: the high frequency band to amplify
    - levels: the depth at which to make the gaussian pyramid
    - alpha: the factor with which to amplify the color

    Returns
    -------
    - The video with amplified color
    """
    gauss = gaussian_video(vid, levels=levels)
    filtered = temporal_ideal_filter(gauss, low, high, fps)
    amplified = alpha * filtered
    return reconstruct_video_g(amplified, vid, levels=levels)

def canvas2numpy(canvas, context):
    """ Grabs the canvas image and converts it to numpy.

    Parameters
    ----------
    - canvas: the canvas element from the DOM.
    - context: the context of the canvas.

    Returns
    -------
    - arr: a numpy array.
    """
    img_data = context.getImageData(0, 0, canvas.width, canvas.height)
    arr = np.frombuffer(img_data.data.to_py(), np.uint8, -1)
    return arr.reshape(canvas.height, canvas.width, 4)[...,:3]

class HeartRateFinder:
    """ A class for finding heart rate of multiple faces.

    Parameters
    ----------
    - num_frames: the number of frames to buffer.
    - frame_width: the width of the window to use for the buffer.
    - frame_height: the height of the window to use for the buffer.
    """

    def __init__(self, num_frames, frame_width, frame_height):
        self.buffer_shape = (num_frames, frame_height, frame_width, 3)
        self.canvas = js.document.querySelector("canvas")
        self.ctx = self.canvas.getContext("2d")
        self.buffers = []
        self.times = []
        self.bpm = []
        self.frame = None

    def collect_frame(self):
        """Collect the frame from the canvas."""
        self.frame = canvas2numpy(self.canvas, self.ctx)

    def get_heart_rate(self, face_id, box, timestamp) -> float:
        """Get the heart rate for a face.

        Parameters
        ----------
        - face_id: the ID for the face we are getting the heartrate for.
        - box: the bounding box around the face (dict with x, y, width, height)
        - timestamp: the timestamp at which the face was collected.

        Returns
        -------
        - the heart rate.
        """
        if face_id >= len(self.buffers):
            self.buffers.append(np.zeros(self.buffer_shape))
            self.times.append(np.zeros((self.buffer_shape[0],)))
            self.bpm.append(np.zeros((self.buffer_shape[0],)))
        _, H, W, _ = self.buffer_shape
        cx = int(box["x"] + (box["width"] / 2))
        cy = int(box["y"] + (box["height"] / 2))
        self.buffers[face_id][:-1] = self.buffers[face_id][1:]
        self.buffers[face_id][-1] = self.frame[cy-H//2:cy+H//2,cx-W//2:cx+W//2]
        self.times[face_id][:-1] = self.times[face_id][1:]
        self.times[face_id][-1] = timestamp
        self.bpm[face_id][:-1] = self.bpm[face_id][1:]
        self.bpm[face_id][-1] = find_heart_rate(
            vid=self.buffers[face_id],
            times=self.times[face_id],
            low=0.8333,
            high=1.0
        )
        return np.mean(self.bpm[face_id])
