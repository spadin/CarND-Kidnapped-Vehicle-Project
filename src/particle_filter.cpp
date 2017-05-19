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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;

  std::default_random_engine gen;
  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  if (!is_initialized)
  {
    for (int i = 0; i < num_particles; i++)
    {
      Particle particle;
      particle.x = N_x(gen);
      particle.y = N_y(gen);
      particle.theta = N_theta(gen);

      particles.push_back(particle);
    }
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    Particle &p = particles[i];
    double x0 = p.x;
    double y0 = p.y;
    double theta0 = p.theta;

    double theta = theta0 + yaw_rate * delta_t;
    double x = x0 + velocity / yaw_rate * (sin(theta) - sin(theta0));
    double y = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta));

    std::normal_distribution<double> N_x(x, std_pos[0]);
    std::normal_distribution<double> N_y(y, std_pos[1]);
    std::normal_distribution<double> N_theta(theta, std_pos[2]);

    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++)
  {
    LandmarkObs &o = observations[i];

    double minDistance = 1.0e100;
    int minId = -1;
    for (int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs p = predicted[j];

      double distance = dist(o.x, o.y, p.x, p.y);
      if (distance < minDistance)
      {
        minDistance = distance;
        minId = p.id;
      }
    }

    o.id = minId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  std::vector<LandmarkObs> landmarks;
  for (int i = 0; i < map_landmarks.landmark_list.size(); i++)
  {
    LandmarkObs l;
    l.x = map_landmarks.landmark_list[i].x_f;
    l.y = map_landmarks.landmark_list[i].y_f;
    l.id = map_landmarks.landmark_list[i].id_i;
    landmarks.push_back(l);
  }

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  double two_sigma_x2 = 2 * sigma_x * sigma_x;
  double two_sigma_y2 = 2 * sigma_y * sigma_y;
  double one_over_2pi_sigma_xy = (1 / (2 * M_PI * sigma_x * sigma_y));

  std::vector<double> nextWeights;

  for (int i = 0; i < num_particles; i++)
  {
    Particle &p = particles[i];
    std::vector<LandmarkObs> transformedObservations;

    for (int i = 0; i < observations.size(); i++)
    {
      LandmarkObs op = observations[i];
      LandmarkObs om;

      if (dist(op.x, op.y, 0, 0) < sensor_range)
      {
        om.id = op.id;
        om.x = op.x * cos(p.theta) - op.y * sin(p.theta) + p.x;
        om.y = op.x * sin(p.theta) + op.y * cos(p.theta) + p.y;

        transformedObservations.push_back(om);
      }
    }
    dataAssociation(landmarks, transformedObservations);

    long double weight = 1.0;

    for (int i = 0; i < transformedObservations.size(); i++)
    {
      LandmarkObs l = transformedObservations[i];

      double x = l.x;
      double y = l.y;
      int idx = l.id - 1;

      double mu_x = map_landmarks.landmark_list[idx].x_f;
      double mu_y = map_landmarks.landmark_list[idx].y_f;

      double x_minus_mu_x2 = (x - mu_x) * (x - mu_x);
      double y_minus_mu_y2 = (y - mu_y) * (y - mu_y);

      long double exponential = -((x_minus_mu_x2 / two_sigma_x2) + (y_minus_mu_y2 / two_sigma_y2));
      long double w = one_over_2pi_sigma_xy * exp(exponential);

      weight *= w;
    }

    nextWeights.push_back(weight);
  }

  weights = nextWeights;
}

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine gen;

  std::discrete_distribution<double> distribution(weights.begin(), weights.end());

  std::vector<Particle> resampledParticles;

  for (int i = 0; i < num_particles; i++)
  {
    particles[i].weight = weights[i];
  }

  for (int i = 0; i < num_particles; i++)
  {
    int index = distribution(gen);
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
}

void ParticleFilter::write(std::string filename)
{
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i)
  {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}
