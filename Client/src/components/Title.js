import React from "react";
import "./styles/styles.css";
import Button from "@material-ui/core/Button";
import { makeStyles } from "@material-ui/core/styles";
import Lottie from 'react-lottie';

const useStyles = makeStyles(theme => ({
  btn: {
    margin: theme.spacing(1),
    width: 350,
    color: "white",
    backgroundColor: "#00796b",
    "&:hover": {
      backgroundColor: "#00796b",
      opacity: 0.8
    },
    fontFamily: "Baloo 2"
  }
}));



function Title() {
  const styles = useStyles();
  return (
    <div>
      <h1 className="title">Accentrix</h1>
      <p className="subtitle">Connect Beyond Borders</p>
    </div>
  );
}
export default Title;
