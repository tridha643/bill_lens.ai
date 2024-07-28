import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './BillForm.css'; // Import the CSS file for styling

const BillForm = () => {
    const [congress, setCongress] = useState('');
    const [billNumber, setBillNumber] = useState('');
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('http://127.0.0.1:5000/fetch-bill', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ congress, bill_num: billNumber }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Network response was not ok');
            }

            const billData = await response.json();
            console.log('API Response:', billData);
            navigate('/dashboard', { state: { billData, congressType: congress, summary: billData.summary, classification: billData.classification } }); // Pass classification
        } catch (error) {
            console.error('Error fetching bill data:', error);
            setError(error.message || 'Failed to fetch bill data. Please try again.');
        }
    };

    return (
        <div className="bill-form-container">
            <h1 className="bill-title">BillLens</h1>
            <div className="container-wrapper">
                <div className="about-text">
                    <h2 className="about-title">About</h2>
                    <p>Recent observations have indicated a decline in general public awareness of bills passed in Congress. 
                        This trend is attributed by many to the overshadowing of some bills by others in greater popularity in 
                        the media and public interest. For Example, surveys by the Pew Research Center indicate that many Americans 
                        are not fully aware of government procedures and specific legislative actions. Furthermore, there is a lack 
                        of awareness among the public regarding which party endorses or proposes particular bills. To address this 
                        issue, we have developed a machine learning model to provide insights into the political leanings associated 
                        with each bill. The model, trained on historical bill data, achieves a relatively high accuracy of 67%. With 
                        the help of this model, we intend to foster greater public participation in the democratic process through 
                        increased awareness.</p>
                </div>
                <div className="image-placeholder">
                    <img src="/flag.png" alt="Placeholder" />
                </div>
            </div>
            <div className="main-content-container">
                <div className="about-container">
                    <p>Enter the Congress number and Bill number to fetch the bill details.</p>
                </div>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <input
                            type="text"
                            value={congress}
                            onChange={(e) => setCongress(e.target.value)}
                            placeholder="Enter Congress Number"
                        />
                    </div>
                    <div className="form-group">
                        <input
                            type="text"
                            value={billNumber}
                            onChange={(e) => setBillNumber(e.target.value)}
                            placeholder="Enter Bill Number"
                        />
                    </div>
                    <button type="submit">Submit</button>
                </form>
                {error && <div className="error-message">{error}</div>}
            </div>

            <div className='call-to-action'>
            <h2>Call to Action: Why You Should Have This App</h2>
            <div className='paragraph'>
                <p> In today's complex and fast-paced political landscape, staying informed 
                about the legislation that shapes our lives can be a daunting task. Our app 
                revolutionizes the way you engage with the political process, offering you a 
                powerful tool to understand the true nature of the bills passing through Congress. 
                Here's why you should have this app</p>
            </div>
            
            <div className='content-tiles' style={{ overflow: 'hidden' }}>
                <div className="content-tile" style={{ overflow: 'hidden' }}>
                    <p><strong>Empower Yourself with Knowledge:</strong><br /> <span style={{ fontSize: '0.7em', textAlign: 'left' }}>Gain a clear, unbiased understanding of any bill in
Congress with our advanced ML model. Know if it's leaning Republican, Democrat, or truly
bipartisan, and get detailed summaries to stay informed on the issues that matter to you.</span></p>
                </div>
                <div className="content-tile" style={{ overflow: 'hidden' }}>
                    <p><strong>Enhance Civic Participation:</strong><br /> <span style={{ fontSize: '0.7em', textAlign: 'left' }}>Make your voice heard with confidence. By understanding the
partisan leanings and sentiments of legislation, you can make more informed decisions when
voting, advocating, or discussing political issues.</span></p>
                </div>
                <div className="content-tile" style={{ overflow: 'hidden' }}>
                    <p><strong>Boost Your Career:</strong><br /> <span style={{ fontSize: '0.7em', textAlign: 'left' }}>Whether you're a policy analyst, lawyer, or advocate, this app provides you
with the insights and data you need to excel in your field. Stay ahead of legislative trends and
understand the political landscape like never before.</span></p>
                </div>
                <div className="content-tile" style={{ overflow: 'hidden' }}>
                    <p><strong>Promote Transparency and Accountability:</strong><br /> <span style={{ fontSize: '0.7em', textAlign: 'left' }}>Hold your representatives accountable. By
revealing the true nature of bills, our app encourages transparency in the legislative process
and fosters a more accountable government.</span></p>
                </div>

                <div className="content-tile" style={{ overflow: 'hidden' }}>
                    <p><strong>Join a Community of Engaged Citizens:</strong><br /> <span style={{ fontSize: '0.7em', textAlign: 'left' }}>Be part of a movement that values informed,
fact-based discourse. Contribute your own insights and annotations, and help improve our
collective understanding of the legislation that impacts us all.</span></p>
                </div>
            </div>

            <p>Don't just observe the political process—be an active participant. Download our app today and
            take the first step towards a more informed and empowered future.</p>
            </div>



            <div className='call-to-action'>
            <h2>Future Technologies</h2>
            <ul className="left-align-text">
                <li><strong>Explainable AI (XAI):</strong>
                    <ul>
                        <li><strong>Concept:</strong> Enhance transparency by explaining how the ML model makes its predictions.</li>
                        <li><strong>Impact:</strong> Users can understand why a bill is classified as leaning towards a particular party, fostering trust and comprehension of AI decisions.</li>
                    </ul>
                </li>
                <li><strong>Sentiment Analysis and Topic Modeling:</strong>
                    <ul>
                        <li><strong>Concept:</strong> Implement advanced NLP techniques to analyze the sentiment and main topics of each bill.</li>
                        <li><strong>Impact:</strong> Provide deeper insights into the bill's content, revealing underlying themes and potential impacts.</li>
                    </ul>
                </li>
                <li><strong>Blockchain for Transparency:</strong>
                    <ul>
                        <li><strong>Concept:</strong> Use blockchain to securely track and verify changes to bill classifications and user interactions.</li>
                        <li><strong>Impact:</strong> Enhance trust in the app's data integrity and decision-making process, making it tamper-proof and transparent.</li>
                    </ul>
                </li>
                <li><strong>Crowdsourced Annotations:</strong>
                    <ul>
                        <li><strong>Concept:</strong> Allow users to provide their own annotations and classifications of bills, which can be used to continuously improve the ML model.</li>
                        <li><strong>Impact:</strong> Engage the community in the analysis process, leading to more accurate and diverse perspectives on bills.</li>
                    </ul>
                </li>
            </ul>
            </div>
                
        </div>
    );
};

export default BillForm;