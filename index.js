
class HeartRateFinder {
    constructor(threshold = 0.6) {
        this.threshold = threshold;
        this.descriptors = [];
        this.pyodide;
    }

    async setupPyodide() {
        const overlay = document.querySelector(".overlay");
        const loadMsg = document.querySelector("#load-message");
        const progress = document.querySelector("#progress");
        progress.style.width = "0%";
        loadMsg.innerHTML = "Loading pyodide...";
        this.pyodide = await loadPyodide({
            lockfileUrl: "repodata.json"
        });
        progress.style.width = "16%";
        loadMsg.innerHTML = "Loading micropip...";
        await this.pyodide.loadPackage("micropip");
        const micropip = this.pyodide.pyimport("micropip");

        progress.style.width = "32%";
        loadMsg.innerHTML = "Loading numpy...";
        await micropip.install("numpy");

        progress.style.width = "48%";
        loadMsg.innerHTML = "Loading scipy...";
        await micropip.install("scipy");

        progress.style.width = "64%";
        loadMsg.innerHTML = "Loading opencv-python...";
        await micropip.install("opencv-python");

        progress.style.width = "80%";
        loadMsg.innerHTML = "Loading heartrate.min.py...";
        const src = await fetch("heartrate.min.py").then(value => value.text());

        progress.style.width = "96%";
        loadMsg.innerHTML = "Running heartrate.min.py...";
        this.pyodide.runPython(src);

        progress.style.width = "100%";
        this.pyodide.runPython(`hr_finder = HeartRateFinder(60, 40, 40)`);
    }

    queryDescriptors(query) {
        for(let i = 0;i < this.descriptors.length;i++) {
            if(faceapi.euclideanDistance(this.descriptors[i], query) < this.threshold) {
                return i;
            }
        }
        this.descriptors.push(query);
        return this.descriptors.length - 1;
    }

    collectFrame() {
        this.pyodide.runPython(`hr_finder.collect_frame()`);
    }

    findHeartRate(faceId, box, timestamp) {
        const pybox = `{"x": ${box.x}, "y": ${box.y}, "width": ${box.width}, "height": ${box.height}}`;
        const hr = this.pyodide.runPython(`hr = hr_finder.get_heart_rate(${faceId}, ${pybox}, ${timestamp / 1000});hr`);
        return hr;
    }
}


async function main() {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    ctx.font = "Bold Arial 3em";
    const textHeight = ctx.measureText("M").width;
    const video = document.createElement("video");
    await faceapi.loadTinyFaceDetectorModel("models");
    await faceapi.loadFaceRecognitionModel("models");
    await faceapi.loadFaceLandmarkTinyModel("models");
    const options = new faceapi.TinyFaceDetectorOptions()

    const heartRateFinder = new HeartRateFinder(0.6);
    await heartRateFinder.setupPyodide();

    function render(timestamp) {

        faceapi.detectAllFaces(video, options).withFaceLandmarks(true).withFaceDescriptors()
        .then(faces => {
            heartRateFinder.collectFrame();
            // Mirror the image without mirroring the text.
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(-1, 1);
            ctx.translate(-canvas.width / 2, -canvas.height / 2);
            ctx.drawImage(video, 0, 0);
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(-1, 1);
            ctx.translate(-canvas.width / 2, -canvas.height / 2);

            for(const face of faces) {
                const faceId = heartRateFinder.queryDescriptors(face.descriptor);
                const box = face.detection.box;

                const hr = heartRateFinder.findHeartRate(faceId, box, timestamp);

                const x = canvas.width - box.x - box.width;
                ctx.strokeStyle = "red";
                ctx.lineWidth = 3;
                ctx.strokeRect(x, box.y, box.width, box.height);
                ctx.fillStyle = "red";
                ctx.fillRect(x, box.y + box.height - textHeight, box.width, textHeight);
                ctx.fillStyle = "white";
                ctx.fillText(`${faceId}: ${hr.toFixed(2)} bpm`, x + 5, box.y + box.height);
            }
        });
        requestAnimationFrame(render);
    }
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
        document.querySelector(".overlay").style.display = "none";
        requestAnimationFrame(render);
    });
}

window.onload = main;
