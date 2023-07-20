import * as React from "react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";

export default function RowRadioButtonsGroup(props) {
  const { title, label1, label2, label3, label4 } = props;
  const [selectedRadioValue, setSelectedRadioValue] = React.useState(label1);

  React.useEffect(() => {
    sessionStorage.setItem(title.replaceAll(" ", "_"), label1);
  }, []);

  const handleChange = (event) => {
    setSelectedRadioValue(event.target.value);
    sessionStorage.setItem(title.replaceAll(" ", "_"), event.target.value);
  };
  return (
    <FormControl>
      <FormLabel id="demo-row-radio-buttons-group-label">{title}</FormLabel>
      <RadioGroup
        row
        aria-labelledby="demo-row-radio-buttons-group-label"
        name="row-radio-buttons-group"
      >
        <FormControlLabel
          value={label1}
          control={
            <Radio
              checked={selectedRadioValue === label1}
              onChange={handleChange}
              value={label1}
              name="radio-buttons"
              inputProps={{ "aria-label": { label1 } }}
            />
          }
          label={label1}
        />
        <FormControlLabel
          value={label2}
          control={
            <Radio
              checked={selectedRadioValue === label2}
              onChange={handleChange}
              value={label2}
              name="radio-buttons"
              inputProps={{ "aria-label": { label2 } }}
            />
          }
          label={label2}
        />
        <FormControlLabel
          value={label3}
          control={
            <Radio
              checked={selectedRadioValue === label3}
              onChange={handleChange}
              value={label3}
              name="radio-buttons"
              inputProps={{ "aria-label": { label3 } }}
            />
          }
          label={label3}
        />
        <FormControlLabel
          value={label4}
          control={
            <Radio
              checked={selectedRadioValue === label4}
              onChange={handleChange}
              value={label4}
              name="radio-buttons"
              inputProps={{ "aria-label": { label4 } }}
            />
          }
          label={label4}
        />
      </RadioGroup>
    </FormControl>
  );
}
