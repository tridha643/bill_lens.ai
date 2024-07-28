import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import BillForm from './components/BillForm';
import BillDashboard from './components/BillDashboard';

function App() {
    return (
        <Router>
            <div className="App">
                <Routes>
                    {/* Route to the main page */}
                    <Route path="/" element={<BillForm />} />
                    {/* Route to the dashboard on submission of form */}
                    <Route path="/dashboard" element={<BillDashboard />} />
                </Routes>
                <footer className="App-footer">
                    <div className="footer-content">
                        <div className="google-form-container">
                            <iframe
                                src="https://docs.google.com/forms/d/e/1FAIpQLSdjqyMGtNWsc6rPZH3Wb5mwTj1Wmh74amPwTONvpg9qJXFJZg/viewform?embedded=true"
                                width="100%"
                                height="418"
                                frameBorder="0"
                                marginHeight="0"
                                marginWidth="0"
                            >
                                Loading…
                            </iframe>
                        </div>
                        <p className="trademark">@ Bri Kirchgessner | Ayen Monasha | Tridhatri Vallamkondu</p>
                    </div>
                </footer>
            </div>
        </Router>
    );
}

export default App;