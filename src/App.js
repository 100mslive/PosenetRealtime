import React, {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import "./App.css";
import { load } from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import JSConfetti from "js-confetti";
import { drawKeypoints, drawSkeleton } from "./utilities";
import { poseSimilarity } from "posenet-similarity";
import { useCounter } from "./useCounter";
import { images } from "./constants";

const jsConfetti = new JSConfetti();
let requestId;
let expectedPose;

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const posenetRef = useRef(null);
  const [distance, setDistance] = useState(0);
  const [maxDistance, setMaxDistance] = useState(distance);
  const { counter, startCounter } = useCounter();
  const [currentImage, setCurrentImage] = useState(-1);
  const [success, setSuccess] = useState(Array(images.length).fill(null));
  const [isPosenetInit, setPoseNetInitState] = useState(false);
  const [deviceId, setDeviceId] = React.useState("");
  const [devices, setDevices] = React.useState([]);

  const handleDevices = React.useCallback((mediaDevices) => {
    const videoDevices = mediaDevices.filter(
      ({ kind }) => kind === "videoinput"
    );
    setDevices(videoDevices);
  }, []);

  /**
   * @param {posenet.PoseNet} net
   */
  const detect = useCallback(async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4 &&
      posenetRef.current
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      const net = posenetRef.current;

      // Make Detections
      const pose = await net.estimateSinglePose(video, {
        flipHorizontal: true,
      });
      if (expectedPose) {
        const data = poseSimilarity(expectedPose, pose, {
          strategy: "cosineSimilarity",
        });
        setDistance(data);
      }
      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);
      requestId = window.requestAnimationFrame(() => detect());
    }
  }, []);
  //  Load posenet
  const runPosenet = useCallback(async () => {
    posenetRef.current = await load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    setPoseNetInitState(true);
    // window.requestAnimationFrame(detect);
  }, []);

  const start = useCallback(async () => {
    if (currentImage === images.length - 1) {
      setCurrentImage(0);
      setSuccess(Array(images.length).fill(null));
    } else {
      setCurrentImage(currentImage + 1);
    }
  }, [currentImage, detect]); //eslint-disable-line

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;
    drawKeypoints(pose["keypoints"], 0.6, ctx);
    drawSkeleton(pose["keypoints"], 0.7, ctx);
  };

  useEffect(() => {
    if (!posenetRef.current) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then(() => {
          navigator.mediaDevices.enumerateDevices().then(handleDevices);
        })
        .catch(console.error);
      runPosenet();
    }
  }, [runPosenet, handleDevices]);

  useEffect(() => {
    if (currentImage !== -1 && imgRef.current.src === images[currentImage]) {
      imgRef.current.onload = async () => {
        setDistance(0);
        setMaxDistance(0);
        cancelAnimationFrame(requestId);
        expectedPose = null;
        startCounter(10);
        detect();
        expectedPose = await posenetRef.current.estimateSinglePose(
          imgRef.current
        );
      };
    }
  }, [currentImage, detect, startCounter]);

  useEffect(() => {
    if (distance > maxDistance) {
      setMaxDistance(distance);
    }
  }, [distance, maxDistance]);

  // once the start counter ends or scores exceeds reset and start break of 5secs
  useEffect(() => {
    if (
      success[currentImage] === null &&
      (counter === 0 || maxDistance >= 0.99)
    ) {
      setSuccess((value) => {
        value[currentImage] = maxDistance >= 0.99;
        return value;
      });
      setDistance(0);
      setMaxDistance(0);
      cancelAnimationFrame(requestId);
    }
  }, [maxDistance, startCounter, success, currentImage]); //eslint-disable-line

  // show the success confetti
  useEffect(() => {
    if (success.every((value) => value)) {
      jsConfetti.addConfetti({ emojis: ["✨"] });
    }
  }, [success]);

  const renderStatus = () => {
    if (currentImage === -1 || success[currentImage] === null) {
      return null;
    }
    return success[currentImage] ? (
      <div className="status-container">
        <div className="status success">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8.84367 19.434C9.30782 19.4734 9.68323 19.3112 9.96914 18.9561L20.6004 6.94447C20.8088 6.67107 20.9085 6.43254 20.9285 6.19609C20.9813 5.5743 20.5798 5.10803 19.9492 5.05455C19.5201 5.01815 19.244 5.1535 18.9464 5.54282L9.0205 16.8293L4.66648 11.4234C4.43272 11.0595 4.17312 10.8964 3.78779 10.8637C3.15724 10.8102 2.66396 11.2182 2.61196 11.8313C2.58893 12.1028 2.66298 12.3737 2.86765 12.6645L7.80527 18.7814C8.0791 19.1927 8.38827 19.3954 8.84367 19.434Z"
              fill="white"
            />
          </svg>
        </div>
        Success!!
      </div>
    ) : (
      <div className="status-container">
        <div className="status error">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="M18.7071 6.70711C19.0976 6.31658 19.0976 5.68342 18.7071 5.29289C18.3166 4.90237 17.6834 4.90237 17.2929 5.29289L12 10.5858L6.70711 5.29289C6.31658 4.90237 5.68342 4.90237 5.29289 5.29289C4.90237 5.68342 4.90237 6.31658 5.29289 6.70711L10.5858 12L5.29289 17.2929C4.90237 17.6834 4.90237 18.3166 5.29289 18.7071C5.68342 19.0976 6.31658 19.0976 6.70711 18.7071L12 13.4142L17.2929 18.7071C17.6834 19.0976 18.3166 19.0976 18.7071 18.7071C19.0976 18.3166 19.0976 17.6834 18.7071 17.2929L13.4142 12L18.7071 6.70711Z"
              fill="white"
            />
          </svg>
        </div>
        Failed
      </div>
    );
  };

  return (
    <div className="container">
      <select
        value={deviceId}
        onChange={(e) => setDeviceId(e.target.value)}
        style={{ visibility: deviceId ? "hidden" : "visible" }}
      >
        <option value="">Select Camera</option>
        {devices.map((value) => {
          return (
            <option key={value.deviceId} value={value.deviceId}>
              {value.label}
            </option>
          );
        })}
      </select>
      {deviceId && (
        <Fragment>
          <div className="centered">
            <Webcam
              ref={webcamRef}
              mirrored
              videoConstraints={{
                frameRate: { max: 15 },
                deviceId: deviceId || "default",
              }}
            />
            <canvas className="canvas" ref={canvasRef} />
          </div>
          <button onClick={start} disabled={counter > -1 || !isPosenetInit}>
            {counter > 0 ? counter : "Start"}
          </button>
          <div
            className="image-score"
            style={{
              visibility: currentImage !== -1 ? "visible" : "hidden",
            }}
          >
            <img
              ref={imgRef}
              alt="pose"
              crossOrigin="anonymous"
              src={images[currentImage]}
              id="pose-match"
              style={{ width: "100%", marginBottom: 8 }}
            />
            <h1>{maxDistance.toFixed(2)}</h1>
            {renderStatus()}
          </div>
        </Fragment>
      )}
    </div>
  );
}

export default App;
