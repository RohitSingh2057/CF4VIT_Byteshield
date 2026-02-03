def process_new_session(new_session_df, historical_df):
    """
    Processes a single new session using historical context.
    Does NOT modify historical_df.
    """

    user_id = new_session_df.iloc[0]["user_id"]

    # Fetch historical context
    user_history = historical_df[historical_df["user_id"] == user_id]

    # Extract baseline
    baseline = compute_user_baseline(user_history)

    # Add deviation features
    enriched_session = add_deviation_features(new_session_df, baseline)

    # Run risk & decision logic
    result = run_decision_engine(enriched_session.iloc[0])

    return result