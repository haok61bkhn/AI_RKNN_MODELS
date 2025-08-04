#include "BaseTrack.h"

int BaseTrack::_count = 0;

BaseTrack::BaseTrack()
    : track_id(0), is_activated(false), state(TrackState::New), curr_feature(0),
      score(0), start_frame(0), frame_id(0), time_since_update(0),
      location(std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()) {}

BaseTrack::BaseTrack(float score)
    : track_id(0), is_activated(false), state(TrackState::New), curr_feature(0),
      score(score), start_frame(0), frame_id(0), time_since_update(0),
      location(std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()) {}

int BaseTrack::end_frame() const { return this->frame_id; }

int BaseTrack::next_id() { return ++_count; }

void BaseTrack::activate() { throw std::runtime_error("NotImplementedError"); }

void BaseTrack::predict() { throw std::runtime_error("NotImplementedError"); }

void BaseTrack::update() { throw std::runtime_error("NotImplementedError"); }

void BaseTrack::mark_lost() { this->state = TrackState::Lost; }

void BaseTrack::mark_removed() { this->state = TrackState::Removed; }

void BaseTrack::reset_count() { _count = 0; }

bool BaseTrack::get_is_activated() const { return this->is_activated; }

TrackState BaseTrack::get_state() const { return this->state; }

float BaseTrack::get_score() const { return this->score; }

int BaseTrack::get_start_frame() const { return this->start_frame; }

int BaseTrack::get_frame_id() const { return this->frame_id; }

int BaseTrack::get_track_id() const { return this->track_id; }