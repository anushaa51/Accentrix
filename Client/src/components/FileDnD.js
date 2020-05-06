import React from "react";
import "./styles/styles.css";
import { makeStyles } from "@material-ui/core/styles";

import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";
import BackupIcon from "@material-ui/icons/Backup";
import Axios from "axios";
import Button from '@material-ui/core/Button';
import GraphicEqIcon from '@material-ui/icons/GraphicEq';

import ReactDropzone from "react-dropzone";
import { Typography } from "@material-ui/core";

const useStyles = makeStyles(theme => ({
  uploadArea: {
    margin: theme.spacing(1),
    minWidth: 120,
    width: 500,
    height: 250,
    border: "1px dashed darkgrey"
  },

  uploadAreaOff: {
    border: "none",
    height: 150,
    width: 500
  }
}));

function FileDndD(props) {
  const classes = useStyles();
  const [droppedFile, setDropppedFile] = React.useState(undefined);
  const [errorText, setErrorText] = React.useState("");
  const refAud = React.useRef(undefined);
  const onDrop = e => {
    e.preventDefault();
    setDropppedFile(e.target.files[0]);
  };

  let inputFile = '';

  const proxyClicker = () => {
    inputFile.click();
  };

  const uploadFiles = () => {
    var formdata = new FormData();
    formdata.append("audio_file", droppedFile);
    formdata.append("from", props.from);
    formdata.append("to", props.to);

    var requestOptions = {
      method: 'POST',
      body: formdata,
      redirect: 'follow'
    };

    fetch("http://localhost:5000/process", requestOptions)
      .then(response => response.json())
      .then(result => {
        if (result.failed) {
          setErrorText(result.reason);
          return;
        }
        console.log(result)
        props.setLoaded(false)
        props.setData(result)
      })
      .catch(error => {
        console.log('error', error);
      });
  };

  return (
    <div>
      {droppedFile && droppedFile.length !== 0 ? (
        <div style={{ marginTop: 60, marginBottom: 10 }}>
          <GraphicEqIcon style={{ fontSize: 64 }} />
          <Typography style={{ fontSize: 16 }}>
            {droppedFile.name}
          </Typography>
          <Typography style={{ fontSize: 12 }}>
            {droppedFile.type} - {(droppedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </Typography>
          {console.log(droppedFile)}
        </div>
      ) : (
          <>
            <div style={{ marginTop: 60, marginBottom: 10 }} onClick={() => proxyClicker()}>
              <BackupIcon style={{ fontSize: 64 }} />
              <Typography style={{ fontSize: 16 }}>
                Click to Upload
                  </Typography>
              <Typography style={{ fontSize: 14, marginTop: 20 }}>
                Please ensure that the audio file format is WAV
                  </Typography>
              <input type="file" className="form-control" name="file" onChange={onDrop} style={{ display: "none" }} ref={input => {
                // assigns a reference so we can trigger it later
                inputFile = input;
              }} />
            </div>
          </>
        )}

      <Button onClick={() => uploadFiles()} disabled={!droppedFile} variant="contained" style={{ width: 505, marginTop: 25 }} color="primary">
        Analyze and Convert MFCCs
            </Button>

      <Typography style={{ fontSize: 14, marginTop: 20 }}>
        {errorText}
      </Typography>

      {errorText.length !== 0 ? <Button onClick={() => window.location.reload()}  style={{ width: 505, marginTop: 25, marginBottom: 25 }} color="primary">
        Try Again
            </Button> : ""}

    </div >
  );
}

export default FileDndD;
