package eegfeat

import "encoding/json"

type Status struct {
	Recordings []Recording `json:"recordings"`
}

type Recording struct {
	Label   string  `json:"label"`
	Summary string  `json:"summary"`
	Stages  []Stage `json:"stages"`
	Next    *Next   `json:"next"`
}

type Stage struct {
	Stage  string `json:"stage"`
	State  string `json:"state"`
	Reason string `json:"reason"`
}

type Next struct {
	Kind   string `json:"kind"`
	Stage  string `json:"stage"`
	Target string `json:"target"`
}

type Gate struct {
	Stage    string  `json:"stage"`
	Parent   string  `json:"parent"`
	ParentID string  `json:"parent_id"`
	FitID    string  `json:"fit_id"`
	Method   string  `json:"method"`
	Field    string  `json:"field"`
	Items    []Item  `json:"items"`
	Spans    []Span  `json:"spans"`
	Duration float64 `json:"duration"`
}

// ID is kept as the bytes Python sent (a channel name or an index) so the
// decision echoes it without a type round trip.
type Item struct {
	ID        json.RawMessage `json:"id"`
	Label     string          `json:"label"`
	Tags      []string        `json:"tags"`
	Score     *float64        `json:"score"`
	Suggested bool            `json:"suggested"`
}

type Span struct {
	Onset       float64 `json:"onset"`
	Duration    float64 `json:"duration"`
	Description string  `json:"description"`
	Suggested   bool    `json:"suggested"`
}

type Decision struct {
	ParentID string
	FitID    string
	Field    string
	IDs      []json.RawMessage
	Spans    []Span
}

// MarshalJSON writes the shape review --decisions validates: parent_id, fit_id
// when the gate had one, the field, and spans for the raw gate alone.
func (d Decision) MarshalJSON() ([]byte, error) {
	out := map[string]any{"parent_id": d.ParentID}
	if d.FitID != "" {
		out["fit_id"] = d.FitID
	}
	if d.Field == "apply" {
		out["apply"] = len(d.IDs) > 0
	} else {
		ids := d.IDs
		if ids == nil {
			ids = []json.RawMessage{}
		}
		out[d.Field] = ids
	}
	if d.Field == "bads" {
		spans := make([]map[string]any, 0, len(d.Spans))
		for _, span := range d.Spans {
			spans = append(spans, map[string]any{
				"onset": span.Onset, "duration": span.Duration, "description": span.Description,
			})
		}
		out["spans"] = spans
	}
	return json.Marshal(out)
}

type Event struct {
	Event   string `json:"event"`
	Subject string `json:"subject"`
	Step    string `json:"step"`
	Level   string `json:"level"`
	Message string `json:"message"`
	Current int    `json:"current"`
	Total   int    `json:"total"`
}

type Result struct {
	Code   int
	Stderr string
}

// Exit 3 is a run that stopped at a review gate, which is what a gated run does.
func (r Result) Failed() bool {
	return r.Code != 0 && r.Code != 3
}
