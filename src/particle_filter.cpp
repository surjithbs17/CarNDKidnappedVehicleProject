/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 60; // may be changed/tuned

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    //Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // This line creates a normal (Gaussian) distribution for x
    // Create normal distributions for y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);


    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_theta;
        Particle p;

        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        p.id = i;
        p.x = sample_x;
        p.y = sample_y;
        p.theta = sample_theta;
        p.weight = 1.0;

        particles.push_back(p);
        weights.push_back(1.0);
    }
    this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    //Set standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    // This line creates a normal (Gaussian) distribution for x
    // Create normal distributions for y and theta
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);


    for (auto &p: particles) {
        //avoid division by zero
        // if yaw rate is zero
        if(std::fabs(yaw_rate) > 0.001) {
            double yaw_dt = yaw_rate*delta_t;
            double v_yaw = velocity/yaw_rate;
            // calculate px, py when yaw rate is zero
            p.x = p.x + v_yaw * (std::sin(p.theta + yaw_dt) - std::sin(p.theta));
            p.y = p.y + v_yaw * (-std::cos(p.theta + yaw_dt) + std::cos(p.theta));
            p.theta = p.theta + yaw_dt;
        }
        else {
            // calculate px, py otherwise
            p.x = p.x + velocity * std::cos(p.theta)*delta_t;
            p.y = p.y + velocity * std::sin(p.theta)*delta_t;
        }
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Finds the predicted measurement that is closest to each observed measurement
    // and assign the observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto &observation : observations) {
        // set minimum distance to a disproportionately large number at first
        double min_dist = 999999.99;
        // set an invalid ID prior to matching
        int map_id = -1;

        for (auto lm_predicted : predicted) {
            // get distance between current/predicted landmarks
            double distance = dist(observation.x, observation.y, lm_predicted.x, lm_predicted.y);

            // find the landmark closest to the current observation
            if (distance < min_dist) {
                min_dist = distance;
                map_id = lm_predicted.id;
            }
        }

        // set the observation's id to the nearest predicted landmark's id
        observation.id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Updates the weights of each particle using a multi-variate Gaussian distribution.
    // You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // these parameters are fixed throughout the loop, so they are calculated once
    double sx = std_landmark[0];
    double sy = std_landmark[1];
    double weights_sum = 0.0;

    for(int i=0; i<num_particles; ++i) {
        // get particle data
        double px = particles[i].x;
        double py = particles[i].y;
        double heading = particles[i].theta;

        //// for each particle: ////

        // convert the observations to global (map) coordinate system
        vector<LandmarkObs> observations_global_coords;
        for (const auto &observation : observations) {
            double trans_x = cos(heading)* observation.x - sin(heading)* observation.y + px;
            double trans_y = sin(heading)* observation.x + cos(heading)* observation.y + py;

            LandmarkObs obs_map{observation.id, trans_x, trans_y};
            observations_global_coords.push_back(obs_map);
        }

        // Check which landmarks are nearby (0-sensor_range)
        std::vector<LandmarkObs> nearby_landmarks;
        for (auto landmark : map_landmarks.landmark_list) {
            // extract landmark info for clarity
            float x_landmark = landmark.x_f;
            float y_landmark = landmark.y_f;
            int id_landmark = landmark.id_i;

            // filter out only landmarks within the sensor range of the current particle
            // if the distance to the landmark is within the circle governed by the sensor_range, it's in!
            if (dist(x_landmark,y_landmark, px, py) <= sensor_range) {
                LandmarkObs landmark_in_range{id_landmark, x_landmark, y_landmark};
                // put all nearby landmarks in a vector
                nearby_landmarks.push_back(landmark_in_range);
            }
        }

        // match observations with known landmarks
        dataAssociation(nearby_landmarks, observations_global_coords);

        // check measurement probability
        // initial weight is 1
        particles[i].weight = 1.0;

        for(auto &observation: observations_global_coords) {
            double x_obs = observation.x, y_obs = observation.y;
            double x_lm, y_lm;

            // get the x,y coordinates of the predicted landmark associated with the current observation
            for (auto landmark : nearby_landmarks) {
                if (landmark.id == observation.id) {
                    x_lm = landmark.x;
                    y_lm = landmark.y;
                }
            }

            // here we have the landmark's and observed X and Y coordinates.
            // calculating weight using a multivariate gaussian
            double dx = x_lm - x_obs;
            double dy = y_lm - y_obs;
            // set weight (importance)
            particles[i].weight *= ( 1/(2*M_PI*sx*sy)) * std::exp( -(std::pow(dx, 2) / (2*std::pow(sx, 2)) +
                                                                (std::pow(dy, 2) / (2*std::pow(sy, 2)))));
            weights[i] = particles[i].weight;
        }
        weights_sum += particles[i].weight;
    }

    for(int i=0; i<num_particles; ++i) {
        particles[i].weight /=weights_sum;
        weights[i] /= weights_sum;
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Resamples particles with replacement with probability proportional to their weight.
    std::vector<Particle> resampled_particles;

    // This link was useful for this part:
    // https://stackoverflow.com/questions/31153610/setting-up-a-discrete-distribution-in-c

    // Using a discrete distribution to resample particles
    std::random_device rd;
    std::default_random_engine gen(rd());

    for (int i=0; i<num_particles; ++i) {
        discrete_distribution<int> index(weights.begin(), weights.end());
        resampled_particles.push_back(particles[index(gen)]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}