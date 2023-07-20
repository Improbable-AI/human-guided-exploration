import Slider from "./Slider";
import Radio from "./Radio";
// import RadioWithImage from "./RadioWithImage";
import React from "react";
import Button from "@mui/material/Button";
//import ImageComponent from "./ImageComponent"
import { createTheme, ThemeProvider } from '@mui/material/styles';
import styled from "styled-components";
import { useRef, useState } from 'react';
import TaskDescription from "./TaskDescription";
import { Player } from 'video-react';
import 'video-react/dist/video-react.css'; // import css
import { Store } from 'react-notifications-component';
import addNotification from 'react-push-notification';

const theme = createTheme({
  palette: {
    left: {
      light: '#757ce8',
      main: '#3f50b5',
      dark: '#002884',
      contrastText: '#fff',
    },
    right: {
      light: '#ff7961',
      main: '#f44336',
      dark: '#ba000d',
      contrastText: '#000',
    },
    dontknow: {
      light: '#808080',
      main: '#808080',
      dark: '#808080',
      contrastText: '#000',
    },
  },
});

function App() {


  const Container = styled.div`
    text-align: center;
  `;

  const Image = styled.img`
    width: 100%;
    height: 100%;
    /* max-width: 640px;
    max-height: 500px; */
  `;

  const Canvas = styled.div`
    margin-top: 50px;
    max-width: 640px;
    max-height: 500px;
    margin: auto;
  `;

  const SliderContainer = styled.div`
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    max-width: 1200px;
    margin: 20px auto;
  `;

  const CustomizedTitle = styled.div`
    background-color: #a31f34;
    color: white;
    padding: 15px;
    margin-bottom: 38px;
    font-size: 1.25rem;
    line-height: inherit;
    text-align: start;
    width: 100%;
  `;

  function checkNotificationPromise() {
    try {
      Notification.requestPermission().then();
    } catch(e) {
      return false;
    }

    return true;
  }

  function askNotificationPermission() {
    // function to actually ask the permissions
    function handlePermission(permission) {
        console.log("enabled")
        var img = '/imgs/logo_dalle.png';
        var text = 'Hi! I need your help to guide me in my learning.';
        var notification = new Notification('HuGE', { body: text, icon: img });
        console.log("notification created")
    }
  
    // Let's check if the browser supports notifications
    if (!('Notification' in window)) {
      console.log("This browser does not support notifications.");
    } else {
      if(checkNotificationPromise()) {
        Notification.requestPermission()
        .then((permission) => {
          handlePermission(permission);
        })
      } else {
        Notification.requestPermission(function(permission) {
          handlePermission(permission);
        });
      }
    }
  }

  function comeBackNotification(){
    addNotification({
      title: 'Warning',
      subtitle: 'This is a subtitle',
      message: 'This is a very long message',
      theme: 'darkblue',
      native: true // when using native, your OS will handle theming.
  });
    //askNotificationPermission()
    console.log("Adding notification")
    
  }

  const base = "http://localhost:8000" 

  const composeImageUrl = () => {
    const questionId = sessionStorage.getItem("questionid")


    let url = `${base}/image?`;
    const items = {
      questionId,
    };

    for (const key in items) {
      url += `${key}=${items[key]}&`;
    }

    return url.slice(0, -1);
  };
  const composeQuestionUrl = (img_index, reset) => {
    const answer = sessionStorage.getItem("answer")
    const questionId = sessionStorage.getItem("questionid")

    let url = `${base}/answer_question?`;
    const items = {
      answer,
      questionId,
    };

    for (const key in items) {
      url += `${key}=${items[key]}&`;
    }

    return url.slice(0, -1);
  };

  const composeVideoUrl = (img_index, reset) => {
    const {
      answer,
    } = sessionStorage;


    let url = `${base}/video?`;
    const items = {
      answer,
    };

    for (const key in items) {
      url += `${key}=${items[key]}&`;
    }

    return url.slice(0, -1);
  };

  const defaultVal = "none"
  const label = "answer"
  const defaultId = -1
  const questionId = "questionid"

  var stop_signal;
  var reset_signal;
  const NUM_COLLECTED_LABELS_PER_BATCH = 30;
  const WAIT_TIME_BETWEEN_LABELLING = 0.5;

  const [value, setV] = React.useState(defaultVal);
  const [imagesrc, setImage] = React.useState("img.png");
  const [q_id, setQuestionId] = React.useState(-1);
  const [waiting, setWaiting] = React.useState(false);
  const [videosrc, setVideo] = React.useState("");
  const [collected_labels, setCollectedLabels] = React.useState(1);

  const MINUTE_MS = 60000;
  const QUERY_VIDEO_FREQ = 1;

  React.useEffect(() => {
    const url = `${base}/start`
    console.log("hi")
    console.log(url)
    fetch(url)//"http://improbable008.csail.mit.edu:8000/start")
    sessionStorage.setItem(label.replaceAll(" ", "_"), defaultVal);
    sessionStorage.setItem(questionId.replaceAll(" ", "_"), defaultId);
    selected("none");

    // const interval = setInterval(() => {
    //   console.log('Fetching video');
    //   fetchVideo()
    // }, QUERY_VIDEO_FREQ*MINUTE_MS);

  //return () => clearInterval(interval); // This represents the unmount function, in which you need to clear your interval to prevent memory leaks.
  }, []);

  const setValue = (newVal) => {
    setV(newVal);
    sessionStorage.setItem(label.replaceAll(" ", "_"), newVal);
  };


  const ImageComponent = ({src, waiting}) => { 
      const ImageStyle = styled.img`
      width: 100%;
      height: 100%;
      /* max-width: 640px;
      max-height: 500px; */
    `;

      const image = useRef(null);
      const [valid, setValid] = useState(true);

      const checkValid = () => {
          if (!image.current.complete || image.current.naturalWidth < 1 || image.current.naturalHeight < 1) setValid(false);
      }

      if (waiting) {
        // schedule new call
        return <div> 
                <br></br>
                <br></br>
                <br></br>
                Thank you for your help! We really appreciate it :) Don't forget to fill the form: <a target="_blank" href="https://docs.google.com/forms/d/e/1FAIpQLSegd0e-tJcFZgX15EeGtubXRRdx0v3p__COHekloLb0BtlbUw/viewform?usp=sf_link">here</a>
                <br></br>
                <br></br>
                <br></br>
            </div>;
      }

      if (valid) {
          return (
              <ImageStyle
                  src={src}
                  onLoad={checkValid}
                  onError={() => setValid(false)}
                  ref={image} 
              ></ImageStyle>
          );
      }

      return <div> 
                <br></br>
                <br></br>
                <br></br>
                Loading ... 
                <br></br>
                <br></br>
                <br></br>
            </div>;
  };

  const VideoComponent = ({src}) => { 
    const Video = styled.video`
    width: 100%;
    height: 100%;
    /* max-width: 640px;
    max-height: 500px; */
  `;

    const video = useRef(src);
    const [valid, setValid] = useState(true);


    return (
      <Player
      playsInline
      src={src}      />
           
    )
 
};

  const selected = (answer) => {
    setImage("")
    setQuestionId(-1)
    setValue(answer);
    imageLoop();
  }



  const fetchVideo = async () => {
    const video_index = sessionStorage.getItem("video_index");
    const videoUrl = composeVideoUrl(video_index, reset);
    sessionStorage.setItem(
      "video_index",
      Number(video_index) + 1 < 60 ? Number(video_index) + 1 : 0
    );
    const res = await fetch(videoUrl);
    console.log("Fetch video", res)
    const videoBlob = await res.blob();
    const videoObjectURL = URL.createObjectURL(videoBlob);
    console.log(videoObjectURL)
    setVideo(videoObjectURL);
  }

  const fetchQuestion = async () => {
    const img_index = sessionStorage.getItem("img_index");
    const questionUrl = composeQuestionUrl(img_index, reset);
    const response = await fetch(questionUrl);
    const blob = await response.blob()
    const new_id = await blob.text()
    sessionStorage.setItem(questionId.replaceAll(" ", "_"), new_id);
    console.log("Fetch new question", new_id)

    // console.log("new image", imageObjectURL);
  };

  const fetchImage = async () => {
    const img_index = sessionStorage.getItem("img_index");
    const imageUrl = composeImageUrl();
    sessionStorage.setItem(
      "img_index",
      Number(img_index) + 1 < 60 ? Number(img_index) + 1 : 0
    );
    const res = await fetch(imageUrl);
    console.log("Fetch image", res)
    const imageBlob = await res.blob();
    const imageObjectURL = URL.createObjectURL(imageBlob);
    console.log(imageObjectURL)
    setImage(imageObjectURL);
    {/*document.getElementById("result_img").src = imageObjectURL;*/}
    // console.log("new image", imageObjectURL);
  };

  const imageLoop = () => {
    requestAnimationFrame(async () => {
      console.log("collected labels pre", collected_labels)
      setCollectedLabels(collected_labels + 1)
      console.log("collected labels", collected_labels)
      if ( collected_labels ==  NUM_COLLECTED_LABELS_PER_BATCH )  {
        console.log("reached max labels")
        setWaiting(true)
        // setTimeout(async () =>{
        //   console.log("Finish waitime and sending notification")
        //   comeBackNotification()
        //   setWaiting(false)
        //   setCollectedLabels(1);
        //   fetchQuestion()
        //   fetchImage();
        // }, WAIT_TIME_BETWEEN_LABELLING*MINUTE_MS)
      } else {
        await fetchQuestion()
        fetchImage();
      }
    });
    // const intervalID = setInterval(fetchImage, 50);
    // sessionStorage.setItem("intervalID", intervalID);
    // sessionStorage.setItem("img_index", 0);
  };

  const stop = () => {
    stop_signal = true;
    // clearInterval(sessionStorage.getItem("intervalID"));
  };

  const reset = async () => {
    reset_signal = true;
    stop_signal = true;
    // stop();
    await fetchQuestion()
    fetchImage(false, true);
    console.log("rest clicked");
  };

  // Create a condition that targets viewports at least 768px wide
  const mediaQuery = window.matchMedia("(max-width: 768px)");
  function handleTabletChange(e) {
    // Check if the media query is true
    if (e.matches) {
      // Then log the following message to the console
      console.log("Media Query Matched!");
    }
  }
  // Register event listener
  mediaQuery.addEventListener("responsive_check", handleTabletChange);
  // Initial check
  handleTabletChange(mediaQuery);

  return (
    <Container>
      <CustomizedTitle>Human-Guided Exploration (HuGE)</CustomizedTitle>
      <h1>
        Breadcrumbs to the Goal: Goal-Conditioned Exploration from Human-in-the-Loop Feedback
      </h1>
      <h3>WEIRD Lab x Improbable AI Lab</h3>
      <h3>University of Washington x Massachusetts Institute of Technology</h3>
      <h4>Kitchen</h4>
      <p>Please first WATCH the video <a target="_blank" href="https://youtu.be/LbDq6Vgatbs"> here</a> </p>
      <p>Please fill in the <a target="_blank" href="https://docs.google.com/forms/d/e/1FAIpQLSegd0e-tJcFZgX15EeGtubXRRdx0v3p__COHekloLb0BtlbUw/viewform?usp=sf_link">form</a></p>
      <p>You will be prompted with {collected_labels-2}/30 images</p>
      <TaskDescription/>
      <Canvas>
        <ImageComponent id="result_img" src={imagesrc} waiting={waiting}/>
        {/*<Image id="result_img" crossOrigin="anonymous" src={"imgs/0.png"} alt="result"></Image>*/}
      </Canvas>

      <ThemeProvider theme={theme}>
        <Button
          variant="outlined"
          color="left"
          onClick={() => selected("left")}
          style={{ marginRight: "20px" }}
        >
          Left
        </Button>
        <Button
          variant="outlined"
          color="dontknow"
          onClick={() => selected("dontknow")}
          style={{ marginRight: "20px" }}
        >
          Don't know
        </Button>
        <Button
          variant="outlined"
          color="right"
          onClick={() => selected("right")}
          style={{ marginRight: "20px" }}
        >
          Right
        </Button>
      </ThemeProvider>
      <br></br>
      <br></br>
      <br></br>



      <br></br>
      <br></br>
      <br></br>
    </Container>
  );
}

//      <Canvas>
//<VideoComponent id="result_video" src={videosrc} />
//</Canvas>
export default App;
