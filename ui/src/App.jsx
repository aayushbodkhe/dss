import { useMemo, useState } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

const defaultVisitForm = {
  patient_id: '',
  patient_name: '',
  patient_age: 45,
  patient_allergies: '',
  visit_id: '',
  visit_date: new Date().toISOString().slice(0, 10),
  chief_complaint: '',
  symptoms: '',
  vitals_text: 'BP: 120/80\nHR: 86',
  labs_text: 'Creatinine: 1.1\neGFR: 65',
  doctor_note: '',
};

function parseKeyValueInput(label, value) {
  const text = (value || '').trim();
  if (!text) return {};

  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const result = {};
  for (const line of lines) {
    const sepIdx = line.indexOf(':');
    if (sepIdx <= 0) {
      throw new Error(`${label} format error: use "key: value" on each line.`);
    }
    const key = line.slice(0, sepIdx).trim();
    const rawValue = line.slice(sepIdx + 1).trim();
    if (!key || !rawValue) {
      throw new Error(`${label} format error: missing key or value in "${line}".`);
    }
    result[key] = rawValue;
  }

  return result;
}

function buildVisitPayload(form) {
  const vitals = parseKeyValueInput('Vitals', form.vitals_text);
  const labs = parseKeyValueInput('Labs', form.labs_text);

  return {
    patient_id: form.patient_id.trim(),
    patient_name: form.patient_name.trim(),
    patient_age: Number(form.patient_age),
    patient_allergies: form.patient_allergies
      .split(',')
      .map((x) => x.trim())
      .filter(Boolean),
    visit_id: form.visit_id.trim(),
    visit_date: form.visit_date,
    chief_complaint: form.chief_complaint.trim(),
    symptoms: form.symptoms
      .split('\n')
      .map((x) => x.trim())
      .filter(Boolean),
    vitals,
    labs,
    doctor_note: form.doctor_note.trim(),
  };
}

async function apiRequest(path, method, token, body) {
  const headers = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data;
}

function App() {
  const [token, setToken] = useState(localStorage.getItem('clinicaldss_token') || '');
  const [username, setUsername] = useState(localStorage.getItem('clinicaldss_user') || '');

  const [signinUser, setSigninUser] = useState('clinician');
  const [signinPass, setSigninPass] = useState('demo123');

  const [visitForm, setVisitForm] = useState(defaultVisitForm);
  const [runResult, setRunResult] = useState(null);
  const [hitlState, setHitlState] = useState(null);
  const [decisionNote, setDecisionNote] = useState('');
  const [editedTreatment, setEditedTreatment] = useState('');

  const [pastPatientId, setPastPatientId] = useState('');
  const [pastVisits, setPastVisits] = useState([]);
  const [timelinePatientId, setTimelinePatientId] = useState('');
  const [timelineEvents, setTimelineEvents] = useState([]);

  const [similarPatientId, setSimilarPatientId] = useState('');
  const [similarVisitId, setSimilarVisitId] = useState('');
  const [similarCases, setSimilarCases] = useState([]);

  const [whatIfDifferential, setWhatIfDifferential] = useState([]);
  const [whatIfTreatment, setWhatIfTreatment] = useState('');
  const [whatIfPrimary, setWhatIfPrimary] = useState('');
  const [whatIfSummary, setWhatIfSummary] = useState('');
  const [newDxName, setNewDxName] = useState('');
  const [newDxProb, setNewDxProb] = useState('0.40');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');

  const signedIn = Boolean(token);

  const trendVisuals = useMemo(() => {
    const alerts = runResult?.trend_alerts || [];
    return alerts.map((a, idx) => {
      const raw = Math.abs(Number(a.delta || 0));
      const width = Math.max(8, Math.min(100, raw * 25));
      return {
        id: `${a.lab}-${idx}`,
        lab: a.lab,
        severity: a.severity,
        direction: a.direction,
        delta: a.delta,
        width,
      };
    });
  }, [runResult]);

  const dashboardStats = useMemo(() => {
    const confidence = Math.round((runResult?.confidence_calibrated || 0) * 100);
    return [
      { label: 'Session User', value: username || 'Guest' },
      { label: 'Last Run Status', value: runResult?.status?.toUpperCase() || 'IDLE' },
      { label: 'Confidence', value: `${confidence}%` },
      { label: 'Critical Findings', value: `${(runResult?.critique_findings || []).filter((f) => f.severity === 'CRITICAL').length}` },
      { label: 'Timeline Events', value: `${timelineEvents.length}` },
      { label: 'Similar Cases', value: `${similarCases.length}` },
    ];
  }, [username, runResult, timelineEvents.length, similarCases.length]);

  async function handleSignin(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const data = await apiRequest('/auth/signin', 'POST', null, {
        username: signinUser,
        password: signinPass,
      });
      setToken(data.token);
      setUsername(data.username);
      localStorage.setItem('clinicaldss_token', data.token);
      localStorage.setItem('clinicaldss_user', data.username);
      setToast('Signed in successfully.');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSignout() {
    setLoading(true);
    setError('');
    try {
      await apiRequest('/auth/signout', 'POST', token);
    } catch (_) {
      // Ignore signout failures and clear local state anyway.
    } finally {
      setToken('');
      setUsername('');
      localStorage.removeItem('clinicaldss_token');
      localStorage.removeItem('clinicaldss_user');
      setRunResult(null);
      setHitlState(null);
      setPastVisits([]);
      setLoading(false);
      setToast('Signed out.');
    }
  }

  async function handleRunAgent(e) {
    e.preventDefault();
    setError('');
    setToast('');
    setRunResult(null);
    setHitlState(null);

    let payload;
    try {
      payload = buildVisitPayload(visitForm);
      if (!payload.patient_id || !payload.visit_id || !payload.chief_complaint || payload.symptoms.length === 0) {
        throw new Error('Please fill patient_id, visit_id, chief complaint, and at least one symptom.');
      }
    } catch (err) {
      setError(err.message);
      return;
    }

    setLoading(true);
    try {
      const data = await apiRequest('/visits/run', 'POST', token, payload);
      setRunResult(data);
      setWhatIfDifferential(data.differential || []);
      setWhatIfTreatment(data.final_treatment || data.proposed_treatment || '');
      setWhatIfPrimary(data.primary_diagnosis || '');
      setWhatIfSummary('');
      setSimilarPatientId(payload.patient_id || '');
      setSimilarVisitId(payload.visit_id || '');
      setTimelinePatientId(payload.patient_id || '');

      if (data.status === 'hitl_required') {
        setEditedTreatment(data.proposed_treatment || '');
        setHitlState({
          review_id: data.review_id,
          hitl_reason: data.hitl_reason,
          critique_findings: data.critique_findings || [],
          differential: data.differential || [],
        });
      } else {
        setToast('Treatment plan created. Episodic memory and care plan updated.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleReviewDecision(action) {
    if (!hitlState?.review_id) return;
    setError('');
    setLoading(true);
    try {
      const data = await apiRequest('/visits/review', 'POST', token, {
        review_id: hitlState.review_id,
        action,
        edited_treatment: action === 'EDIT' ? editedTreatment : undefined,
        clinician_note: decisionNote || undefined,
      });
      setHitlState(null);

      if (data.status === 'flagged_for_specialist_review') {
        setToast('Flagged for specialist review.');
      } else {
        setRunResult((prev) => ({
          ...prev,
          status: 'completed',
          final_treatment: data.final_treatment,
          care_plan: data.care_plan,
        }));
        setToast('HITL approved. Treatment plan finalized and memory updated.');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadPastVisits() {
    setError('');
    setToast('');
    if (!pastPatientId.trim()) {
      setError('Enter a patient ID to load past visits.');
      return;
    }
    setLoading(true);
    try {
      const data = await apiRequest(`/patients/${encodeURIComponent(pastPatientId.trim())}/visits`, 'GET', token);
      setPastVisits(data.visits || []);
      setToast(`Loaded ${data.visits?.length || 0} visit(s).`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadTimeline() {
    setError('');
    if (!timelinePatientId.trim()) {
      setError('Enter a patient ID to load timeline.');
      return;
    }
    setLoading(true);
    try {
      const data = await apiRequest(`/patients/${encodeURIComponent(timelinePatientId.trim())}/timeline`, 'GET', token);
      setTimelineEvents(data.events || []);
      setToast(`Loaded timeline with ${data.events?.length || 0} event(s).`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadSimilarCases() {
    setError('');
    if (!similarPatientId.trim()) {
      setError('Enter patient ID to search similar cases.');
      return;
    }
    setLoading(true);
    try {
      const patient = encodeURIComponent(similarPatientId.trim());
      const visitParam = similarVisitId.trim() ? `?visit_id=${encodeURIComponent(similarVisitId.trim())}` : '';
      const data = await apiRequest(`/patients/${patient}/similar${visitParam}`, 'GET', token);
      setSimilarCases(data.similar_cases || []);
      setToast(`Found ${data.similar_cases?.length || 0} similar case(s).`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function recalculateWhatIf(nextDifferential) {
    if (!runResult) return;
    setLoading(true);
    setError('');
    try {
      const data = await apiRequest('/visits/differential/what-if', 'POST', token, {
        patient_id: visitForm.patient_id,
        chief_complaint: visitForm.chief_complaint,
        symptoms: visitForm.symptoms.split('\n').map((x) => x.trim()).filter(Boolean),
        current_primary: runResult.primary_diagnosis || '',
        current_treatment: runResult.final_treatment || runResult.proposed_treatment || '',
        current_differential: nextDifferential,
        add_candidates: [],
        remove_diagnoses: [],
      });
      setWhatIfDifferential(data.updated_differential || []);
      setWhatIfPrimary(data.new_primary || '');
      setWhatIfTreatment(data.suggested_treatment || '');
      setWhatIfSummary(data.explainability_summary || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function addDifferentialCandidate() {
    const diagnosis = newDxName.trim();
    const probability = Number(newDxProb);
    if (!diagnosis) {
      setError('Enter diagnosis name to add.');
      return;
    }
    if (Number.isNaN(probability) || probability < 0 || probability > 1) {
      setError('Probability must be a number between 0 and 1.');
      return;
    }
    const next = [...whatIfDifferential, { diagnosis, probability, reasoning: 'Added by clinician.' }];
    setWhatIfDifferential(next);
    setNewDxName('');
    await recalculateWhatIf(next);
  }

  async function removeDifferentialCandidate(name) {
    const target = (name || '').toLowerCase();
    const next = whatIfDifferential.filter((d) => (d.diagnosis || '').toLowerCase() !== target);
    setWhatIfDifferential(next);
    if (next.length === 0) {
      setWhatIfPrimary('');
      setWhatIfTreatment('');
      setWhatIfSummary('Differential list is empty. Add a diagnosis to continue.');
      return;
    }
    await recalculateWhatIf(next);
  }

  function updateVisitField(key, value) {
    setVisitForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand-block">
          <span className="eyebrow">Clinical Decision Support Platform</span>
          <h1>ClinicalDSS Control Center</h1>
          <p>Stateful agent workspace for timeline intelligence, explainability, and human-guided decisions.</p>
        </div>
        {signedIn && (
          <div className="userbox">
            <span>Signed in as {username}</span>
            <button onClick={handleSignout} className="btn btn-outline">Sign Out</button>
          </div>
        )}
      </header>

      {signedIn && (
        <section className="hero-band">
          {dashboardStats.map((item) => (
            <article className="stat-card" key={item.label}>
              <p className="stat-label">{item.label}</p>
              <h3>{item.value}</h3>
            </article>
          ))}
        </section>
      )}

      {error && <div className="alert error">{error}</div>}
      {toast && <div className="alert success">{toast}</div>}

      {!signedIn ? (
        <section className="signin-card">
          <h2>Sign In</h2>
          <form onSubmit={handleSignin} className="form-grid narrow">
            <label>
              Username
              <input value={signinUser} onChange={(e) => setSigninUser(e.target.value)} />
            </label>
            <label>
              Password
              <input type="password" value={signinPass} onChange={(e) => setSigninPass(e.target.value)} />
            </label>
            <button disabled={loading} className="btn" type="submit">{loading ? 'Signing in...' : 'Sign In'}</button>
          </form>
        </section>
      ) : (
        <main className="home-grid">
          <section className="card">
            <h2>Add Patient Visit</h2>
            <p className="muted">Capture clinician input, run ClinicalDSS flow, and generate treatment/care plan.</p>
            <form onSubmit={handleRunAgent} className="form-grid">
              <label>
                Patient ID
                <input value={visitForm.patient_id} onChange={(e) => updateVisitField('patient_id', e.target.value)} />
              </label>
              <label>
                Patient Name
                <input value={visitForm.patient_name} onChange={(e) => updateVisitField('patient_name', e.target.value)} />
              </label>
              <label>
                Age
                <input type="number" value={visitForm.patient_age} onChange={(e) => updateVisitField('patient_age', e.target.value)} />
              </label>
              <label>
                Allergies (comma-separated)
                <input value={visitForm.patient_allergies} onChange={(e) => updateVisitField('patient_allergies', e.target.value)} />
              </label>
              <label>
                Visit ID
                <input value={visitForm.visit_id} onChange={(e) => updateVisitField('visit_id', e.target.value)} />
              </label>
              <label>
                Visit Date
                <input type="date" value={visitForm.visit_date} onChange={(e) => updateVisitField('visit_date', e.target.value)} />
              </label>
              <label className="full">
                Chief Complaint
                <input value={visitForm.chief_complaint} onChange={(e) => updateVisitField('chief_complaint', e.target.value)} />
              </label>
              <label className="full">
                Symptoms (one per line)
                <textarea rows={4} value={visitForm.symptoms} onChange={(e) => updateVisitField('symptoms', e.target.value)} />
              </label>
              <label>
                Vitals
                <textarea
                  rows={4}
                  placeholder={'BP: 120/80\nHR: 86\nTemp: 98.4 F'}
                  value={visitForm.vitals_text}
                  onChange={(e) => updateVisitField('vitals_text', e.target.value)}
                />
                <span className="field-help">Use one line per item in key:value format.</span>
              </label>
              <label>
                Labs
                <textarea
                  rows={4}
                  placeholder={'Creatinine: 1.1\neGFR: 65\nHbA1c: 7.2'}
                  value={visitForm.labs_text}
                  onChange={(e) => updateVisitField('labs_text', e.target.value)}
                />
                <span className="field-help">Simple format only, no JSON braces or quotes needed.</span>
              </label>
              <label className="full">
                Clinician Notes
                <textarea rows={3} value={visitForm.doctor_note} onChange={(e) => updateVisitField('doctor_note', e.target.value)} />
              </label>
              <button disabled={loading} className="btn" type="submit">
                {loading ? 'Running ClinicalDSS...' : 'Run ClinicalDSS Agent'}
              </button>
            </form>
          </section>

          <section className="card">
            <h2>View Patient Past Visits</h2>
            <p className="muted">Load episodic memory snapshots written by the agent.</p>
            <div className="row">
              <input
                placeholder="Patient ID"
                value={pastPatientId}
                onChange={(e) => setPastPatientId(e.target.value)}
              />
              <button disabled={loading} onClick={loadPastVisits} className="btn btn-outline">Load Visits</button>
            </div>

            <div className="visit-list">
              {pastVisits.length === 0 ? (
                <p className="muted">No visits loaded yet.</p>
              ) : (
                pastVisits.map((v, idx) => (
                  <article key={`${v.visit_id}-${idx}`} className="visit-item">
                    <h4>{v.visit_id} - {v.date}</h4>
                    <p><strong>Complaint:</strong> {v.chief_complaint}</p>
                    <p><strong>Primary:</strong> {v.primary_diagnosis || 'N/A'}</p>
                    <p><strong>Treatment:</strong> {v.final_treatment || 'N/A'}</p>
                  </article>
                ))
              )}
            </div>
          </section>

          <section className="card">
            <h2>Patient Timeline View</h2>
            <p className="muted">Chronological view of visits, vitals, labs, meds, and alerts.</p>
            <div className="row">
              <input
                placeholder="Patient ID"
                value={timelinePatientId}
                onChange={(e) => setTimelinePatientId(e.target.value)}
              />
              <button disabled={loading} onClick={loadTimeline} className="btn btn-outline">Load Timeline</button>
            </div>
            <div className="timeline-list">
              {timelineEvents.length === 0 ? (
                <p className="muted">No timeline loaded yet.</p>
              ) : (
                timelineEvents.map((evt, idx) => (
                  <article key={`${evt.visit_id}-${evt.type}-${idx}`} className="timeline-item">
                    <div className="timeline-meta">
                      <strong>{evt.date || 'N/A'}</strong>
                      <span className="pill timeline-pill">{(evt.type || '').toUpperCase()}</span>
                    </div>
                    <p><strong>{evt.title}:</strong> {evt.detail || 'N/A'}</p>
                  </article>
                ))
              )}
            </div>
          </section>

          <section className="card fullwidth">
            <h2>Patient Similarity Search</h2>
            <p className="muted">Find similar past cases and review outcomes.</p>
            <div className="form-grid">
              <label>
                Patient ID
                <input value={similarPatientId} onChange={(e) => setSimilarPatientId(e.target.value)} />
              </label>
              <label>
                Visit ID (optional)
                <input value={similarVisitId} onChange={(e) => setSimilarVisitId(e.target.value)} />
              </label>
            </div>
            <button disabled={loading} onClick={loadSimilarCases} className="btn btn-outline">Find Similar Cases</button>
            <div className="similar-list">
              {similarCases.length === 0 ? (
                <p className="muted">No similar cases loaded yet.</p>
              ) : (
                similarCases.map((c, idx) => (
                  <article key={`${c.patient_id}-${c.visit_id}-${idx}`} className="visit-item">
                    <h4>{c.patient_id} / {c.visit_id} ({Math.round((c.similarity_score || 0) * 100)}%)</h4>
                    <p><strong>Complaint:</strong> {c.chief_complaint || 'N/A'}</p>
                    <p><strong>Diagnosis:</strong> {c.primary_diagnosis || 'N/A'}</p>
                    <p><strong>Outcome:</strong> {c.outcome_summary || 'N/A'}</p>
                    <p><strong>Matched Signals:</strong> {(c.matched_signals || []).join(', ') || 'N/A'}</p>
                  </article>
                ))
              )}
            </div>
          </section>

          <section className="card fullwidth">
            <h2>Agent Execution</h2>
            {!runResult ? (
              <p className="muted">Run a visit to see progress, trend detection, critique flags, and treatment/care plan.</p>
            ) : (
              <>
                <div className="status-row">
                  <span className={`pill ${runResult.status}`}>{runResult.status.toUpperCase()}</span>
                  <span>Primary: {runResult.primary_diagnosis || 'N/A'}</span>
                  <span>Confidence: {Math.round((runResult.confidence_calibrated || 0) * 100)}%</span>
                </div>

                <h3>Progress</h3>
                <ul className="progress-list">
                  {(runResult.progress || []).map((step, idx) => (
                    <li key={`${step.node}-${idx}`}>
                      <strong>{step.node}</strong>
                      <span>{step.status}</span>
                      <time>{new Date(step.timestamp).toLocaleTimeString()}</time>
                    </li>
                  ))}
                </ul>

                <h3>Trend Detection</h3>
                <p>{runResult.trend_summary}</p>
                {trendVisuals.length > 0 ? (
                  <div className="trend-bars">
                    {trendVisuals.map((t) => (
                      <div key={t.id} className="trend-row">
                        <span>{t.lab}</span>
                        <div className="bar-wrap">
                          <div
                            className={`bar ${t.severity.toLowerCase()}`}
                            style={{ width: `${t.width}%` }}
                            title={`${t.direction || 'N/A'} delta ${t.delta || 0}`}
                          />
                        </div>
                        <small>{t.severity}</small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No trend alerts for this visit.</p>
                )}

                <h3>Critique Findings</h3>
                {(runResult.critique_findings || []).length === 0 ? (
                  <p className="muted">No critique findings.</p>
                ) : (
                  <ul className="finding-list">
                    {runResult.critique_findings.map((f, idx) => (
                      <li key={`${f.type}-${idx}`} className={f.severity?.toLowerCase()}>
                        [{f.severity}] {f.type}: {f.message}
                      </li>
                    ))}
                  </ul>
                )}

                {runResult.status === 'completed' && (
                  <>
                    <h3>Treatment Plan</h3>
                    <p>{runResult.final_treatment || 'N/A'}</p>
                    <h3>Care Plan</h3>
                    <pre className="note-box">{runResult.care_plan || 'N/A'}</pre>
                  </>
                )}

                <h3>Explainability Mode</h3>
                <div className="explain-box">
                  <p>{runResult.explainability?.summary || runResult.diagnosis_reasoning || 'No explainability summary available.'}</p>
                  <p>
                    <strong>Confidence:</strong>{' '}
                    {Math.round(((runResult.explainability?.confidence?.calibrated ?? runResult.confidence_calibrated) || 0) * 100)}%
                    {' '}
                    ({runResult.explainability?.confidence?.label || 'N/A'})
                  </p>
                  <p><strong>Guidelines Used:</strong></p>
                  {(runResult.explainability?.guidelines_used || []).length === 0 ? (
                    <p className="muted">No guideline evidence returned.</p>
                  ) : (
                    <ul className="finding-list">
                      {runResult.explainability.guidelines_used.map((g, idx) => (
                        <li key={`gl-${idx}`}>
                          [{g.id}] {g.title} - {g.excerpt}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <h3>Differential Diagnosis Builder</h3>
                <p className="muted">Add or remove diagnoses and regenerate treatment suggestion instantly.</p>
                <div className="dx-builder">
                  {(whatIfDifferential || []).length === 0 ? (
                    <p className="muted">No differential available yet. Run a visit first.</p>
                  ) : (
                    <ul className="diff-list">
                      {whatIfDifferential.map((d, idx) => (
                        <li key={`whatif-${idx}`} className="dx-row">
                          <span>{Math.round((d.probability || 0) * 100)}% - {d.diagnosis}</span>
                          <button className="btn btn-outline" type="button" onClick={() => removeDifferentialCandidate(d.diagnosis)}>
                            Remove
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}

                  <div className="form-grid">
                    <label>
                      Add Diagnosis
                      <input placeholder="e.g. COPD exacerbation" value={newDxName} onChange={(e) => setNewDxName(e.target.value)} />
                    </label>
                    <label>
                      Probability (0-1)
                      <input value={newDxProb} onChange={(e) => setNewDxProb(e.target.value)} />
                    </label>
                  </div>
                  <button className="btn btn-outline" type="button" onClick={addDifferentialCandidate} disabled={loading || !runResult}>
                    Add And Recalculate
                  </button>

                  <div className="whatif-result">
                    <p><strong>New Primary:</strong> {whatIfPrimary || 'N/A'}</p>
                    <p><strong>Suggested Treatment:</strong> {whatIfTreatment || 'N/A'}</p>
                    <p className="muted">{whatIfSummary || 'Modify the differential list to generate a new treatment recommendation.'}</p>
                  </div>
                </div>
              </>
            )}
          </section>
        </main>
      )}

      {hitlState && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>HITL Review Required</h3>
            <p><strong>Reason:</strong> {hitlState.hitl_reason}</p>

            <h4>Critique Findings</h4>
            {(hitlState.critique_findings || []).length === 0 ? (
              <p className="muted">No critique findings attached.</p>
            ) : (
              <ul className="finding-list">
                {hitlState.critique_findings.map((f, idx) => (
                  <li key={`m-${idx}`} className={f.severity?.toLowerCase()}>
                    [{f.severity}] {f.type}: {f.message}
                  </li>
                ))}
              </ul>
            )}

            <h4>Differential</h4>
            <ul className="diff-list">
              {(hitlState.differential || []).map((d, idx) => (
                <li key={`d-${idx}`}>{Math.round((d.probability || 0) * 100)}% - {d.diagnosis}</li>
              ))}
            </ul>

            <label>
              Clinician Note
              <textarea rows={2} value={decisionNote} onChange={(e) => setDecisionNote(e.target.value)} />
            </label>

            <label>
              Edited Treatment (used only if EDIT)
              <textarea rows={3} value={editedTreatment} onChange={(e) => setEditedTreatment(e.target.value)} />
            </label>

            <div className="modal-actions">
              <button className="btn btn-outline" disabled={loading} onClick={() => handleReviewDecision('REJECT')}>
                Reject
              </button>
              <button className="btn btn-outline" disabled={loading} onClick={() => handleReviewDecision('EDIT')}>
                Approve with Edit
              </button>
              <button className="btn" disabled={loading} onClick={() => handleReviewDecision('APPROVE')}>
                Approve
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
