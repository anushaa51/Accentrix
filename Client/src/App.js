import React from "react";
import logo from "./logo.svg";
import "./App.css";
import Title from "./components/Title";
import AccentSelector from "./components/AccentSelector";
import FileDndD from "./components/FileDnD";
import Grid from "@material-ui/core/Grid";
import { MuiThemeProvider, createMuiTheme } from "@material-ui/core/styles";
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';


const theme = createMuiTheme({
  typography: {
    fontFamily: ['"Open Sans"'].join(",")
  }
});

function App() {
  const [disableAnalyze, setDisableAnalyze] = React.useState(false);
  const [to, setTo] = React.useState("");
  const [from, setFrom] = React.useState("");
  const [loaded, setLoaded] = React.useState(true);
  const [data, setData] = React.useState({});

  return (
    <MuiThemeProvider theme={theme}>
      <div className="App">
        <header className="App-header">
          <Title />
          {loaded ? (<div>
            <AccentSelector setTo={setTo} setFrom={setFrom} />
            <Grid
              container
              direction="column"
              justify="center"
              alignItems="center"
            >
              <Grid item>
                <FileDndD setLoaded={setLoaded} setButtonDisable={setDisableAnalyze} to={to} from={from} setData={setData} />
              </Grid>
            </Grid>
          </div>) : (
              <div>
                <img src={`data:image/png;base64, ${data.mfcc_input}`} />
                <img src={`data:image/png;base64, ${data.mfcc_output}`} />

                <Typography style={{ fontSize: 14, fontFamily: 'Oxanium' }}>Classification Before Conversion :</Typography>
                <Typography style={{ color: 'green' }}>{data.cbc}</Typography>
                <Typography style={{ marginTop: 15, fontSize: 14, fontFamily: 'Oxanium' }}>Classification After Conversion :</Typography>
                <Typography style={{ color: 'green' }}>{data.cac}</Typography>

                <Button onClick={() => window.location.reload()} style={{ width: 505, marginTop: 25, marginBottom: 25 }} color="primary">
                  Try A Different Audio File
                </Button>

                <Typography style={{ marginTop: 10, marginBottom: 5, fontSize: 25, fontFamily: 'Oxanium' }}>Model Performances</Typography>

                <Typography style={{ marginTop: 15, fontSize: 14, fontFamily: 'Oxanium' }}>Classification Accuracy of MFCC Classifier (on Test Data) :</Typography>
                <Typography style={{ color: 'teal' }}>{data.mcaoc}</Typography>
                <Typography style={{ marginTop: 15, fontSize: 14, fontFamily: 'Oxanium' }}>Mean Classification of Validation Data Before Conversion :</Typography>
                <Typography style={{ color: 'teal' }}>{data.mcbc}</Typography>
                <Typography style={{ marginTop: 15, fontSize: 14, fontFamily: 'Oxanium' }}>Mean Classification of Validation Data After Conversion : </Typography>
                <Typography style={{ color: 'teal' }}>{data.mcac}</Typography>
                <Typography style={{ marginTop: 15, fontSize: 14, fontFamily: 'Oxanium' }}>MFCC Converter Accuracy (on Test Data) :</Typography>
                <Typography style={{ color: 'teal', marginBottom: 25 }}>{data.ca}</Typography>

                {/* Comment Below upto next comment if you do not want to include training graphs */}

                <Typography style={{ marginBottom: 5, fontSize: 40, fontFamily: 'Oxanium' }}>Training Performance</Typography>

                <img src={`data:image/png;base64, ${data.classifier1}`} style={{width:638, height:398}} />
                <img src={`data:image/png;base64, ${data.classifier2}`} style={{width:638, height:398}} />
                <Typography style={{ marginTop: 5, marginBottom: 40, fontSize: 25, fontFamily: 'Oxanium' }}>Classifier</Typography>

                <img src={`data:image/png;base64, ${data.converter1}`} />
                <img src={`data:image/png;base64, ${data.converter2}`} />
                <Typography style={{ marginTop: 5, marginBottom: 40, fontSize: 25, fontFamily: 'Oxanium' }}>Converter</Typography>



              </div>)}
        </header>
      </div>
    </MuiThemeProvider>
  );
}

export default App;
