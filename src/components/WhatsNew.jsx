import React from 'react';
import { Container, Box } from '@mui/material';
import './WhatsNew.css';

const WhatsNew = () => {
  return (
    <Container
      maxWidth="lg"
      sx={{ height: '100vh', overflow: 'hidden', padding: '0 !important' }}
    >
      <Box
        className="whats-new-page"
        sx={{
          overflowY: 'auto',
          paddingTop: '80px'  // Add space at the top to prevent overlap
        }}
      >
        <div className="announcement-card">
          <h2>Subject: MOST IMPORTANT:</h2>
          <p>
            Please USE EARPHONES in the Backend sessions.
            <span className="icon">🔥</span>
          </p>
          <p className="date">June 27, 2024, 6:25 p.m.</p>
        </div>
        <div className="announcement-card">
          <h2>Subject: IMPORTANT:</h2>
          <p>
            Assignments miss mt kro.
            <span className="icon">😇</span>
          </p>
          <p className="date">June 24, 2024, 9:13 p.m.</p>
        </div>
        <div className="announcement-card">
          <h2>Subject: Imp <span className="icon">🔥</span>:</h2>
          <p>
            AI Solver is in the house Guyzzz.... Check it out and FEEDBACK is important.
          </p>
          <p className="date">June 24, 2024, 9:13 p.m.</p>
        </div>
      </Box>
    </Container>
  );
};

export default WhatsNew;