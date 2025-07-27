import React from 'react';
import { AppBar, Toolbar, IconButton, Box, useTheme, useMediaQuery } from '@mui/material';
import NotificationsIcon from '@mui/icons-material/Notifications';
import HelpOutlineIcon from '@mui/icons-material/ChatRounded';
import MenuIcon from '@mui/icons-material/Menu';
import { useNavigate } from 'react-router-dom';

const Navbar = ({ handleDrawerToggle }) => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm')); // Identify mobile view
    const navigate = useNavigate();

    const handleNotificationClick = () => {
        navigate('/WhatsNew');
    };

    return (
        <AppBar
            position="absolute"
            color="transparent"
            style={{ boxShadow: 'none', top: 0, right: 0 }}
        >
            <Toolbar
                sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginRight: isMobile ? '24px' : '72px',
                    padding: '8px 16px',
                }}
            >
                {/* Menu Button for Mobile View */}
                {isMobile && (
                    <IconButton
                        edge="start"
                        aria-label="menu"
                        onClick={handleDrawerToggle} // Trigger sidebar toggle
                        sx={{ color: 'white', marginTop: '30px' }}
                    >
                        <MenuIcon />
                    </IconButton>
                )}

                {/* Right-side icons (Notification and Help) */}
                <Box sx={{ display: 'flex', gap: '8px', marginLeft: 'auto', marginTop: '30px' }}>
                    {/* Notification Icon */}
                    <Box
                        sx={{
                            backgroundColor: 'rgb(245,245,245,0.9)',
                            borderRadius: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                    >
                        <IconButton color="inherit" onClick={handleNotificationClick}>
                            <NotificationsIcon />
                        </IconButton>
                    </Box>

                    {/* Help Icon */}
                    <Box
                        sx={{
                            backgroundColor: 'rgb(245,245,245,0.9)',
                            borderRadius: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                    >
                        <IconButton color="inherit">
                            <HelpOutlineIcon />
                        </IconButton>
                    </Box>
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Navbar;
