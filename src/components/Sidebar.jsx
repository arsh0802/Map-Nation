
import React, { useState } from 'react';
import { List, ListItem, ListItemIcon, ListItemText, IconButton, useTheme, useMediaQuery } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import LibraryBooksIcon from '@mui/icons-material/LibraryBooks';
import PersonIcon from '@mui/icons-material/Person';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import MenuIcon from '@mui/icons-material/Menu';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import FeedbackIcon from '@mui/icons-material/Feedback';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = ({ open, handleDrawerToggle }) => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

    // State to manage active list item
    const [activeIndex, setActiveIndex] = useState(null);

    const handleListItemClick = (index) => {
        // Toggle the active item
        if (activeIndex === index) {
            setActiveIndex(null); // Collapse if already active
        } else {
            setActiveIndex(index); // Set active item
        }
    };

    return (
        <>
            {/* Sidebar Content */}
            <div className={`sidebar-content ${isMobile && !open ? 'hidden' : ''}`}>
                <div className="logo-container">
                    <img src="Vector(2).png" alt="Logo" className="logo-image" />
                </div>

                <div className="sidebar-list-container">
                    <List className="sidebar-list">
                        {[ // Sidebar Menu Items
                            { text: 'My Dashboard', icon: <DashboardIcon className="white-icon" />, path: '/Dashboard' },
                            { text: 'My Roadmaps', icon: <LibraryBooksIcon className="white-icon" />, path: '/MyRoadmaps' },
                            { text: "What's New", icon: <NewReleasesIcon className="white-icon" />, path: '/WhatsNew' },
                            { text: 'Feedback', icon: <FeedbackIcon className="white-icon" />, path: '/feedback' },
                            { text: 'My Profile', icon: <PersonIcon className="white-icon" />, path: '/my-profile' },
                            { text: 'Logout', icon: <ExitToAppIcon className="white-icon" />, path: '/' },
                        ].map((item, index) => (
                            <ListItem
                                button
                                key={index}
                                component={Link}
                                to={item.path}
                                className={`list-item ${activeIndex === index ? 'active' : ''}`}
                                onClick={() => handleListItemClick(index)}  // Handle click to toggle active state
                            >
                                <ListItemIcon className="list-item-icon">{item.icon}</ListItemIcon>
                                <ListItemText primary={item.text} className="list-item-text" />
                            </ListItem>
                        ))}
                    </List>
                </div>
            </div>

            
        </>
    );
};

export default Sidebar;
