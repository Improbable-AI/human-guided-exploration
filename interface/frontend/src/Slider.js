import * as React from "react";
import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import MuiInput from "@mui/material/Input";

const Input = styled(MuiInput)`
  width: 60px;
`;

export default function InputSlider(props) {
  const { label, upper, lower, defaultVal, step } = props;

  const [value, setV] = React.useState(defaultVal);

  React.useEffect(() => {
    sessionStorage.setItem(label.replaceAll(" ", "_"), defaultVal);
  }, []);

  const setValue = (newVal) => {
    setV(newVal);
    sessionStorage.setItem(label.replaceAll(" ", "_"), newVal);
  };

  const handleSliderChange = (event, newValue) => {
    setValue(newValue);
  };

  const handleInputChange = (event) => {
    setValue(event.target.value === "" ? "" : Number(event.target.value));
  };

  const handleBlur = () => {
    if (value < lower) {
      setValue(lower);
    } else if (value > upper) {
      setValue(upper);
    }
  };

  return (
    <Box sx={{ minWidth: 250 }}>
      <Typography id="input-slider" gutterBottom>
        {label}
      </Typography>
      <Grid container spacing={2} alignItems="center">
        <Grid item></Grid>
        <Grid item xs>
          <Slider
            value={typeof value === "number" ? value : lower}
            onChange={handleSliderChange}
            aria-labelledby="input-slider"
            max={upper}
            min={lower}
            default={defaultVal}
            step={step}
          />
        </Grid>
        <Grid item>
          <Input
            value={value}
            size="small"
            onChange={handleInputChange}
            onBlur={handleBlur}
            inputProps={{
              step: step,
              min: lower,
              max: upper,
              type: "number",
              "aria-labelledby": "input-slider",
            }}
          />
        </Grid>
      </Grid>
    </Box>
  );
}
