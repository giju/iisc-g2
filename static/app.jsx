const { TextField, FormControl, InputLabel, Select, MenuItem, RadioGroup, CircularProgress, FormControlLabel, Radio, Button, Typography, Slider } = MaterialUI;
 
function SurveyForm() {

    const [formData, setFormData] = React.useState({
        professionType: '',
        profession: '',
        name: '',
        pressure: 2,
        age:26,
        degree: '',
        satisfaction: 2,
        dietaryHabits: '',
        gender:'Male',
        sleepDuration: 8,
        timeSpent: 6,
        financialStress:2,
        suicidalThoughts: '',
        familyHistory: '',
    });

    const submitDisabled = Object.keys(formData).some(name => formData[name] == '') ;
    const [inflight,setInflight] = React.useState(null)

    const [confData, setConfData] = React.useState({
        degree: [],
        city: [],
        profession: [],
        diet: []
    })

    const [result, setResult] = React.useState(null)

    React.useEffect(() => {
        fetch('/api/options').then(resp => {
            return resp.json()
        }).then(data => { 
            setConfData(data);
        })
    }, [])

    const handleChange = (event) => {
        const { name, value } = event.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        setInflight('pending')
        fetch('/api/assess',    {
            method:'POST',
            body:  JSON.stringify(formData),
            headers: {
                accept: 'application.json',
                'content-Type': 'application/json'
            },
        }).then(resp => {
            return resp.json()
        }).then(data => {
            console.log(data)
            setResult(data.content)
            setInflight(null)
        }) 
    };

    return (
        <div style={{ padding: '20px', maxWidth: '600px', margin: 'auto', background: "rgba(255,255,255,0.92)" }}>
            <Typography variant="h5" style={{ marginBottom: '20px' }}>Improve Your mental health</Typography>
            {(result == null && inflight == null) && <form onSubmit={handleSubmit}>
                <Typography variant="h6" style={{ marginBottom: '20px' }}>Take this small survey to get insights on your mental health</Typography>
                <TextField
                    label="Your Name"
                    name="name"
                    fullWidth
                    variant="outlined"
                    margin="normal"
                    onChange={handleChange}
                />
 
                    <label>Your gender </label>
                    <select title="gender" className="w3-input" name="gender" value={formData.gender} onChange={handleChange}>
                        {['Male','Female'].map(item => {
                            return <option value={item}>{item}</option>
                        })}
                    </select> 
                    <label>Which city are you from ?</label>
                    <select  title="select your city" className="w3-select" name="city" value={formData.city} onChange={handleChange}>
                        <option value="">Select</option>
                        {confData.city.map(item => {
                            return <option value={item}>{item}</option>
                        })}
                    </select> 
 
                    <label>What is your educational qualification</label>
                    <select title="Degree" name="degree" className="w3-input" value={formData.degree} onChange={handleChange}>
                        <option value="">Select</option>
                        {confData.degree.map(item => {
                            return <option value={item}>{item}</option>
                        })}
                    </select> 
 
                    <label>How healthy are your dietary habits</label>
                    <select title="Dietary Habits" className="w3-input" name="dietaryHabits" value={formData.dietaryHabits} onChange={handleChange}>
                        <option value="">Select</option>
                        {confData.diet.map(item => {
                            return <option value={item}>{item}</option>
                        })}
                    </select> 

                <Typography variant="subtitle1" gutterBottom>
                    How old are you (Age in years)
                </Typography>
                <Slider
                    name="age"
                    value={formData.age}
                    onChange={(event, value) => setFormData({ ...formData, age: value })}
                    step={1}
                    marks
                    min={12}
                    max={80}
                    valueLabelDisplay="auto"
                />

                <Typography variant="subtitle1" gutterBottom>
                    How many hour do you sleep every day ?
                </Typography>
                <Slider
                    name="sleepDuration"
                    value={formData.sleepDuration}
                    onChange={(event, value) => setFormData({ ...formData, sleepDuration: value })}
                    step={0.5}
                    marks
                    min={0}
                    max={12}
                    valueLabelDisplay="auto"
                />

                <FormControl component="fieldset" style={{ marginBottom: '20px' }}>
                    <Typography variant="subtitle">Are you a Working Professional or Student</Typography>
                    <RadioGroup name="professionType" onChange={handleChange}>
                        <FormControlLabel value="Working Professional" control={<Radio defaultCheked />} label="Working Professional" />
                        <FormControlLabel value="Student" control={<Radio />} label="Student" />
                    </RadioGroup>
                </FormControl>


                {formData.professionType != 'Student' && <div> 
                        <InputLabel>Profession</InputLabel>
                        <select title="select profession" className="w3-input" name="profession" value={formData.profession} onChange={handleChange}>
                            <option value="">Select</option>
                            {confData.profession.map(item => {
                                return <option value={item}>{item}</option>
                            })}
                        </select> 


                    <Typography variant="subtitle1" gutterBottom>
                        Work Pressure  (1 = Low, 5 = High)
                    </Typography>
                    <Slider
                        name="pressure"
                        value={formData.pressure}
                        onChange={(event, value) => setFormData({ ...formData, pressure: value })}
                        step={1}
                        marks
                        min={1}
                        max={5}
                        valueLabelDisplay="auto"
                    />


                    <Typography variant="subtitle1" gutterBottom>
                        How many hours your spent at work  ?
                    </Typography>
                    <Slider
                        name="timeSpent"
                        value={formData.timeSpent}
                        onChange={(event, value) => setFormData({ ...formData, timeSpent: value })}
                        step={1}
                        marks
                        min={1}
                        max={18}
                        valueLabelDisplay="auto"
                    />
                    <Typography variant="subtitle1" gutterBottom>
                        Job Satisfaction  (1 = Low, 5 = High)
                    </Typography>
                    <Slider
                        name="satisfaction"
                        value={formData.satisfaction}
                        onChange={(event, value) => setFormData({ ...formData, satisfaction: value })}
                        step={1}
                        marks
                        min={1}
                        max={5}
                        valueLabelDisplay="auto"
                    />

                </div> }
                { formData.professionType == 'student'  && <div>
                    <TextField
                        label="CGPA"
                        name="cgpa"
                        type="number"
                        fullWidth
                        variant="outlined"
                        margin="normal"
                        min={0}
                        max={10}
                        onChange={handleChange}
                    />
                    <Typography variant="subtitle1" gutterBottom>
                        Academic Pressure  (1 = Low, 5 = High)
                    </Typography>
                    <Slider
                        name="pressure"
                        value={formData.pressure}
                        onChange={(event, value) => setFormData({ ...formData, pressure: value })}
                        step={1}
                        marks
                        min={1}
                        max={5}
                        valueLabelDisplay="auto"
                    />

                    <Typography variant="subtitle1" gutterBottom>
                        How many hours your you spent on study  ?
                    </Typography>
                    <Slider
                        name="timeSpent"
                        value={formData.timeSpent}
                        onChange={(event, value) => setFormData({ ...formData, timeSpent: value })}
                        step={1}
                        marks
                        min={1}
                        max={18}
                        valueLabelDisplay="auto"
                    />

                    <Typography variant="subtitle1" gutterBottom>
                        Study Satisfaction (1 = Low, 5 = High)
                    </Typography>
                    <Slider
                        name="satisfaction"
                        value={formData.satisfaction}
                        onChange={(event, value) => setFormData({ ...formData, satisfaction: value })}
                        step={1}
                        marks
                        min={1}
                        max={5}
                        valueLabelDisplay="auto"
                    />
                </div>}

                {formData.professionType != '' && <div>
                <FormControl component="fieldset" style={{ marginBottom: '20px' }}>
                    <Typography variant="subtitle1">Do you have Family History of Mental Illness</Typography>
                    <RadioGroup name="familyHistory" onChange={handleChange}>
                        <FormControlLabel value="Yes" control={<Radio />} label="Yes" />
                        <FormControlLabel value="No" control={<Radio />} label="No" />
                    </RadioGroup>
                </FormControl>

                <Typography variant="subtitle1" gutterBottom>
                    How do you rate your Financial Stress  (1 = Low, 5 = High)
                </Typography>
                <Slider
                    name="financialStress"
                    value={formData.financialStress}
                    onChange={(event, value) => setFormData({ ...formData, financialStress: value })}
                    step={1}
                    marks
                    min={1}
                    max={5}
                    valueLabelDisplay="auto"
                />
                <FormControl component="fieldset" style={{ marginBottom: '20px' }}>
                    <Typography variant="subtitle1">Have you ever had suicidal thoughts?</Typography>
                    <RadioGroup name="suicidalThoughts" onChange={handleChange}>
                        <FormControlLabel value="Yes" control={<Radio />} label="Yes" />
                        <FormControlLabel value="No" control={<Radio />} label="No" />
                    </RadioGroup>
                </FormControl>
                </div>}

                <Button disabled={submitDisabled} type="submit" variant="contained" color="primary" fullWidth>Submit</Button>
            </form>}
            {inflight == 'pending' && <div style={{margin:'200px 20px',textAlign:'center'}}>
                        <CircularProgress style={{margin:'20px'}}/>     <br/>
                        Please wait while we assess your response
            </div>}
            {result != null && <div className="result">
            <div dangerouslySetInnerHTML={{__html: marked.parse(result) }}></div> 
            
            <Button style={{margin:'0px 20px'}} onClick={e => setResult(null)} variant="outlined">Back to Survey</Button>
            </div>}
            <Typography variant="subtitle1" style={{color:'#AAA',fontSize:'12px',textAlign:'center'}}>This is a capstone project developed for the IISC Machine learning course by group2 CCE Aug 2024</Typography>
        </div>
    );
}
ReactDOM.render(<SurveyForm />, document.getElementById('root'));