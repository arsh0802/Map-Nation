import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './MyProfile.css';

const MyProfile = () => {
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // State to hold profile data
    const [profileData, setProfileData] = useState({
        name: '',
        university: '',
        email: '',
        phone: '',
        linkedIn: '',
        gitHub: '',
        website: '',
        avatar: ''
    });

    // Fetch the user's profile data from the backend
    useEffect(() => {
    const fetchProfileData = async () => {
        setLoading(true);
      

      try {
        const token = await localStorage.getItem('token');
        console.log(token)

                if (token === null) {
                    navigate('/login');
                    return;
                }

        const response = await fetch('http://localhost:5000/api/profile/me', {
          method: 'GET',
          headers: {
           'auth-token':
           token,
          },
        });
        console.log(response.data);

        if (response.ok) {
          const data =  response.json();
          setProfileData({
            name: `${data.firstName} ${data.lastName}`,
            university: data.university || 'N/A',
            email: data.email,
            phone: data.contact || 'N/A',
            linkedIn: data.linkedIn || '',
            gitHub: data.gitHub || '',
            website: data.website || '',
            avatar: data.avatar || '/avatar.jpg'
          });
        } else {
          const errorData = await response.json();
          console.error('Failed to fetch profile data:', errorData?.message || 'Unknown error');
        }
      } catch (error) {
        console.error('Error fetching profile data:', error.message);
      }
    };

    fetchProfileData();
  }, []);

  const handleUpdateProfile = () => {
    navigate('/EditProfile');
  };

    

    return (
        profileData && (
        <div className="profile-page-container">
            <div className="profile-card-container">
                {/* Profile Section */}
                <div className="profile-section">
                    <img src={profileData.avatar} alt="Profile Avatar" />
                    <h2>{profileData.name}</h2>
                    <p>{profileData.university}</p>
                </div>

                {/* Details Section */}
                <div className="details-section">
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>Name</span>
                            <span style={{ fontWeight: 'bold', textShadow: '2px 2px 2px rgba(0, 0, 0, 0.2)' }}>{profileData.name}</span>
                        </div>
                    </div>
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>Email</span>
                            <span style={{ fontWeight: 'bold', textShadow: '2px 2px 2px rgba(0, 0, 0, 0.2)' }}>{profileData.email}</span>
                        </div>
                    </div>
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>Phone</span>
                            <span style={{ fontWeight: 'bold', textShadow: '2px 2px 2px rgba(0, 0, 0, 0.2)' }}>{profileData.phone}</span>
                        </div>
                    </div>
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>LinkedIn</span>
                            <span>
                                <a href={profileData.linkedIn} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 'bold', color: '#0073b1' }}>
                                    LinkedIn Profile
                                </a>
                            </span>
                        </div>
                    </div>
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>GitHub</span>
                            <span>
                                <a href={profileData.gitHub} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 'bold', textShadow: '2px 2px 2px rgba(0, 0, 0, 0.2)' }}>
                                    GitHub Profile
                                </a>
                            </span>
                        </div>
                    </div>
                    <div className="detail-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontWeight: 'bold', color: 'rgb(0, 0, 0)', textShadow: '4px 4px 4px rgba(0, 0, 0, 0.3)' }}>Website</span>
                            <span>
                                <a href={profileData.website} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 'bold', color: '#1e88e5' }}>
                                    Personal Website
                                </a>
                            </span>
                        </div>
                    </div>
                    <button onClick={handleUpdateProfile} className="update-button">
                        Update Profile
                    </button>
                </div>
            </div>
        </div>)
    );
};

export default MyProfile;
