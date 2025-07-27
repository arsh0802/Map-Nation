import React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const PercentageCircle = ({ value }) => {
    return (
        <Box display="flex" flexDirection="column" alignItems="center" position="relative">
            {/* Circular white background */}
            <Box 
                position="absolute"
                top="50%"
                left="50%"
                sx={{
                    width: 80,
                    height: 80,
                    backgroundColor: 'rgb(0,0,0,0.1)',
                    borderRadius: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 1,
                }}
            />
            {/* CircularProgress on top of the white background */}
            <CircularProgress
                color="textPrimary"
                variant="determinate"
                value={value}
                size={80}
                sx={{ position: 'relative', zIndex: 2 }}
            />
            {/* Percentage text in the center */}
            <Typography
                variant="h6"
                color="white"
                sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 3
                }}
            >
                {value}%
            </Typography>
        </Box>
    );
};

export default PercentageCircle;
