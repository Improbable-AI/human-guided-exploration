import * as React from "react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";

export default function RowRadioButtonsGroup(props) {
  const { title, label1, label2, label3, label4 } = props;
  const [selectedValue, setSelectedValue] = React.useState(label1);

  React.useEffect(() => {
    sessionStorage.setItem(title.replaceAll(" ", "_"), label1);
  }, []);

  const handleChange = (event) => {
    setSelectedValue(event.target.value);
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
          labelPlacement="top"
          value={label1}
          control={
            <Radio
              checked={selectedValue === label1}
              onChange={handleChange}
              value={label1}
              name="radio-buttons"
              inputProps={{ "aria-label": { label1 } }}
            />
          }
          label={<img src="imgs/0.png" width="100" height="100"></img>}
        />
        <FormControlLabel
          labelPlacement="top"
          value={label2}
          control={
            <Radio
              checked={selectedValue === label2}
              onChange={handleChange}
              value={label2}
              name="radio-buttons"
              inputProps={{ "aria-label": label2 }}
            />
          }
          label={<img src="imgs/1.png" width="100" height="100"></img>}
        />
        <FormControlLabel
          labelPlacement="top"
          value={label3}
          control={
            <Radio
              checked={selectedValue === label3}
              onChange={handleChange}
              value={label3}
              name="radio-buttons"
              inputProps={{ "aria-label": label3 }}
            />
          }
          label={<img src="imgs/2.png" width="100" height="100"></img>}
        />
        <FormControlLabel
          labelPlacement="top"
          value={label4}
          control={
            <Radio
              checked={selectedValue === label4}
              onChange={handleChange}
              value={label4}
              name="radio-buttons"
              inputProps={{ "aria-label": label4 }}
            />
          }
          label={<img src="imgs/3.png" width="100" height="100"></img>}
        />
      </RadioGroup>
    </FormControl>
  );
}
