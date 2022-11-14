import React, { useCallback, useEffect, useRef, useState } from "react";
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
  const [deviceId, setDeviceId] = React.useState({});
  const [devices, setDevices] = React.useState([]);

  const handleDevices = React.useCallback(
    mediaDevices => {
      const videoDevices = mediaDevices.filter(({ kind }) => kind === "videoinput");
      setDevices(videoDevices);
      setDeviceId(videoDevices[0].deviceId);
    },
    []
  );

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
      navigator.mediaDevices.getUserMedia({ video: true }).then(() => {
        navigator.mediaDevices.enumerateDevices().then(handleDevices);
      }).catch(console.error);
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

  return (
    <div className="container">
      <select value={deviceId} onChange={(e) => console.log(e.target.value)}>
        {devices.map(value => {
          return (<option value={value.deviceId}>{value.label}</option>)
        })}
      </select>
      <div className="centered">
        <Webcam ref={webcamRef} mirrored videoConstraints={{ frameRate: { max: 15 }, deviceId: deviceId || 'default'}}/>
        <canvas className="canvas" ref={canvasRef} />
      </div>
      <button onClick={start} disabled={counter > -1 || !isPosenetInit}>
        {counter > 0 ? counter : "Start"}
      </button>
      <div
        className="image-score"
        style={{
          visibility: counter > -1 ? "visible" : "hidden",
        }}
      >
        <img
          ref={imgRef}
          alt="pose"
          crossOrigin="anonymous"
          src={images[currentImage]}
          id="pose-match"
          style={{ width: "100%" }}
        />
        <h1>{maxDistance.toFixed(2)}</h1>
      </div>
    </div>
  );
}

export default App;
