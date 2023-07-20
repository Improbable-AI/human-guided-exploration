import * as React from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import MuiInput from "@mui/material/Input";
import styled from "styled-components";
import { Container } from "@mui/system";
import { Player } from 'video-react';

const Input = styled(MuiInput)`
  width: 60px;
`;

const theme = {
    blue: {
      default: "#3f51b5",
      hover: "#283593"
    },
    pink: {
      default: "#e91e63",
      hover: "#ad1457"
    },
    gray: {
        hover: "#828282",
        default: "#999DA0"
    }
  };
  
const Button = styled.button`
    background-color: ${(props) => theme[props.theme].default};
    color: white;
    padding: 5px 15px;
    border-radius: 5px;
    outline: 0;
    text-transform: uppercase;
    margin: 10px 0px;
    cursor: pointer;
    box-shadow: 0px 2px 2px lightgray;
    transition: ease background-color 250ms;
    alignment: center;
    &:hover {
      background-color: ${(props) => theme[props.theme].hover};
    }
    &:disabled {
      cursor: default;
      opacity: 0.7;
    }
  `;
  
  Button.defaultProps = {
    theme: "gray"
  };
  

export default function TaskDescription(props) {
  const {task} = props;

  const [value, setV] = React.useState(false);

  const toggle = () => {
    if (value) {
        setV(false)
    }else{
        setV(true)
    }
  };
  let task_text = ""
  if (task == "pointmass_rooms"){
    task_text = "In this task, you are controlling a robot to move around a house. In the image below you can see an aerial view of the home, this has 4 rooms delimited by the walls in black. You have to help the robot (blue/red dot) to get to the room on the top-right. Click on the Red button if the red robot is closer to the goal (green dot), click the blue button if the blue robot is closer to the goal, and click \"don't know\" if you are unsure or if you see a problem."
  } else if (task == "block stacking"){
    task_text = "Control a robot to stack three blocs on top of each other in the following order: red, green and blue, as you see in the video below. Click on the left/right button if you think the left/right picture respectively is closer to achieving the desired goal, and click \"don't know\" if you are unsure or if you see a problem."
  } else {
    task_text = "<p>Order: 1. Open the slider (top right)</p> \n 2. Open the microwave (bottom left) \n 3. Open the cabinet (top left) as you see in the video below. \n Click on the left/right button if you think the left/right picture respectively is closer to achieving the task, and click \"don't know\" if you are unsure or if you see a problem."
  }


  if (value) {
    return (
    <Container>
        <Button onClick={toggle}>Hide task description</Button>
        <div>
          <p>Order:</p>
          <p>1. Open the slider (top right)</p>
          <p>2. Open the microwave (bottom left)</p> 
          <p>3. Open the cabinet (top left) </p>  
          <p>See the video below. </p>
          <p>Click on the left/right button if you think the left/right picture respectively is closer to achieving the task, and click "don't know" if you are unsure or if you see a problem. </p>
        </div>
        <br></br>
        <br></br>
        <br></br>
        <Player
        playsInline
        src="/videos/kitchen3elems.mp4"      />
        <br></br>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
      </Container>
     )
  }
  else {
    return (
        <Container>
            <Button onClick={toggle}>Show task description</Button>
        </Container>
       )
  }
}
