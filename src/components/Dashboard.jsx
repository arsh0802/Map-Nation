import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import { Container, Grid, Card, CardContent, Typography, Box, Button } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import PercentageCircle from './PercentageCircle'; // Adjust the path as needed
import './Dashboard.css'; // Import the new CSS file

// Create a custom theme with Georgia font
const theme = createTheme({
  typography: {
    fontFamily: 'Georgia, serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
    background: 'none', // Decrease background opacity
        },
      },
    },
    MuiTypography: {
      styleOverrides: {
        root: {
          fontFamily: 'Georgia, serif',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none', // Remove uppercase transform
          backgroundColor: 'black',
        },
      },
    },
  },
});

const Dashboard = () => {
  const [studentsEnrolledPercentage, setStudentsEnrolledPercentage] = useState(0);
  const [resourcesAccessedPercentage, setResourcesAccessedPercentage] = useState(0);
  const navigate = useNavigate(); // Initialize useNavigate

  // Simulate fetching data
  useEffect(() => {
    const fetchData = async () => {
      const fetchedData = {
        studentsEnrolled: 75,
        resourcesAccessed: 45,
      };

      setStudentsEnrolledPercentage(fetchedData.studentsEnrolled);
      setResourcesAccessedPercentage(fetchedData.resourcesAccessed);
    };

    fetchData();
  }, []);

  const batchData = [
    { no: 1, name: 'java', days: '15', hours: '2' },
    { no: 2, name: 'c++', days: '20', hours: '3' },
    { no: 3, name: 'javascript', days: '12', hours: '1 ' },
    { no: 4, name: 'node', days: '20', hours: '3' },
    { no: 5, name: 'database', days: '5', hours: '1' }
  ];

  const handleResumeClick = () => {
    // Redirect to the resume page with batchId
    navigate(`/resume`); // Assuming you have a route set up for resuming based on the batchId
  }
  return (
    <ThemeProvider theme={theme}>
      {/* Background Video */}
      <div className="dashboard-container">
        <div className="overlay">
          <Container maxWidth="lg" style={{ marginTop: '20px' }}>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4}>
                <Card>
                  <CardContent className='card-1'>
                    <Box flex={1}>
                      <Typography variant="h6">Tasks Completed</Typography>
                    </Box>
                    <Box>
                      <PercentageCircle value={studentsEnrolledPercentage} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <Card>
                  <CardContent className='card-2'>
                    <Box flex={1}>
                      <Typography variant="h6">Overall Progress</Typography>
                    </Box>
                    <Box>
                      <PercentageCircle value={resourcesAccessedPercentage} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={4}>
                <Card>
                  <CardContent className='card-3'>
                    <Box flex={1}>
                      <Typography variant="h6">Average Task Scores</Typography>
                    </Box>
                    <Box>
                      <PercentageCircle value={studentsEnrolledPercentage} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent className='card-4'>
                    <Typography variant="h6" gutterBottom className="heading">
                      <b><u>Continue Where You Left !!</u></b>
                    </Typography>
                    <Grid container className="tableHeaderContainer">
                      <Grid item xs={1}>
                        <Typography variant="body1" className="tableHeader">
                          S No.
                        </Typography>
                      </Grid>
                      <Grid item xs={3}>
                        <Typography variant="body1" className="tableHeader">
                          Name
                        </Typography>
                      </Grid>
                      <Grid item xs={3}>
                        <Typography variant="body1" className="tableHeader">
                          days
                        </Typography>
                      </Grid>
                      <Grid item xs={2}>
                        <Typography variant="body1" className="tableHeader">
                          Action
                        </Typography>
                      </Grid>
                      <Grid item xs={3}>
                        <Typography variant="body1" className="tableHeader">
                          hours
                        </Typography>
                      </Grid>
                    </Grid>
                    {batchData.map((batch, index) => (
                      <Grid container key={index} alignItems="center" className="tableRow">
                        <Grid item xs={1}>
                          <Typography variant="body1" className="tableContent">
                            {batch.no}
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body1" className="tableContent">
                            {batch.name}
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body1" className="tableContent">
                            {batch.days}
                          </Typography>
                        </Grid>
                        <Grid item xs={2}>
                          <Button 
                           onClick={() => handleResumeClick(batch.no)} // Handle button click
                           variant="contained" 
                           color="primary">
                           Resume
                          </Button>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body1" className="tableContent">
                            {batch.hours}
                          </Typography>
                        </Grid>
                      </Grid>
                    ))}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Container>
        </div>
      </div>
    </ThemeProvider>
  );
};

export default Dashboard;