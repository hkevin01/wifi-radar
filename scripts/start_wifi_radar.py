                # Run through neural network
                with torch.no_grad():
                    # Encode CSI data
                    encoded_features = encoder(amplitude_tensor, phase_tensor)
                    
                    # Estimate pose
                    keypoints, confidence, hidden_state = pose_estimator(encoded_features, hidden_state)
                    
                    # Convert to numpy for visualization
                    keypoints_np = keypoints[0].cpu().numpy()
                    confidence_np = confidence[0].cpu().numpy()
                    
                    # Detect people
                    people = pose_estimator.detect_people(keypoints, confidence)
                    
                    # Update dashboard and RTMP stream with the first detected person
                    if len(people) > 0:
                        first_person = people[0]
                        
                        # Update dashboard
                        dashboard.update_data(
                            pose_data=first_person,
                            confidence_data=first_person['confidence'],
                            csi_data=(amplitude, phase)
                        )
                        
                        # Update RTMP stream
                        rtmp_streamer.update_frame(
                            pose_data=first_person,
                            confidence_data=first_person['confidence']
                        )
                        
                # Sleep briefly to prevent CPU overuse
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
    
    # Start processing thread
    proc_thread = threading.Thread(target=processing_thread)
    proc_thread.daemon = True
    proc_thread.start()
    
    # Start dashboard (this will block until the dashboard is closed)
    logger.info(f"Starting dashboard on port {args.dashboard_port}")
    try:
        dashboard.run(debug=args.debug, port=args.dashboard_port)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up
        logger.info("Stopping system components")
        csi_collector.stop()
        rtmp_streamer.stop()
        
    logger.info("WiFi-Radar system stopped")

if __name__ == "__main__":
    main()
