# Anomaly-Detection

The subject of the project focuses on identifying unusual movements in e-commerce stores for the following indicators:
1. The percentage of customers who entered the site and made a purchase
2. Average order value
3. Customer traffic - how many customers entered the website.
4. Shopping cart abandonment rate

The purpose of the project is to create a basis for a system that can alert the customers of Blyp, which analyzes data for E-commerce stores and provides them with insights. We will alert you when unusual traffic has occurred in their store, we will indicate by what measure and why. The system takes into account factors such as day of the week, quarter, holiday and the index itself so that we do not receive a false alarm.

Many companies use Machine Learning (ML) and Deep Learning (DL) - to identify anomalies. Google for example uses ML to detect anomalies in its data centers, networks and applications. Netflix uses anomaly detection to characterize their streaming service and identify any unusual activity that may affect video quality or user experience.
These are just a few examples of companies using ML/DL for anomaly detection. The use of this technology is becoming more and more common as more companies seek to automate their monitoring and detection processes to improve efficiency and accuracy.

The project is based on an untagged Blyp database. It is important to note that the database is static from the end of June, meaning that there is no flow of new data into it. Blyp does not use ML models to alert customers to unusual traffic. We define the anomaly based on the data for each index and explain it with the help of additional features in the report (holiday, time, revenues, the state of the industry in the same index, etc.). We do this using several ML models which are used for unsupervised problems.

The reports were created for the days defined as exceptions in each index in June, because this is the most recent information we have. In an ideal situation, we would like to create a report using real-time data, but since the information is not available to us, we had to use the data available to us. We created the report using a webapp called Streamlit.
