import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { Box } from '@mui/material';
import './App.css'; // Import the CSS globally for applicable pages
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import MyRoadmaps from './components/MyRoadmaps';
import MyProfile from './components/MyProfile';
import Feedback from './components/Feedback';
import WhatsNew from './components/WhatsNew';
import EditProfile from './components/EditProfile';
import Topics from './components/LanguageTopics';
import Landing from './components/Landing';
import Login from './components/Login';
import Register from './components/SignUp';
import Resume from './components/resume';

const AppContent = () => {
  const location = useLocation();
  const [fullData, setFullData] = useState(null);
  const [open, setOpen] = useState(false); // State to manage sidebar open/close

  const handleDrawerToggle = () => {
    setOpen((prevOpen) => !prevOpen); // Toggle sidebar state
  };

  // Pages that don't need Sidebar, Navbar, and App.css
  const noSidebarAndNavbar = ['/login', '/register', '/'];

  // Determine if App.css should be applied
  const appClass = !noSidebarAndNavbar.includes(location.pathname)? 'app-wrapper' : '';

  return (
    <Box sx={{ display: 'flex' }} className={appClass}>
      {/* Conditionally render Sidebar and Navbar */}
      {!noSidebarAndNavbar.includes(location.pathname) && (
        <>
          <Sidebar open={open} handleDrawerToggle={handleDrawerToggle} />
          <Navbar handleDrawerToggle={handleDrawerToggle} />
        </>
      )}

      <Box component="main" sx={{ flexGrow: 1}}>
        {/* Content Routes */}
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/Dashboard" element={<Dashboard />} />
          <Route path="/MyRoadmaps" element={<MyRoadmaps setFullData={setFullData} />} />
          <Route path="/feedback" element={<Feedback />} />
          <Route path="/my-profile" element={<MyProfile />} />
          <Route path="/WhatsNew" element={<WhatsNew />} />
          <Route path="/EditProfile" element={<EditProfile />} />
          <Route path="/topics" element={<Topics fullData={fullData} />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/resume" element={<Resume />} />
        </Routes>
      </Box>
    </Box>
  );
};

const App = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;
