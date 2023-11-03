import pandas as pd
import streamlit as st
import os
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pickle
pd.options.mode.chained_assignment = None



def set_bg_url():
    """
    A function to unpack an image from url and set as background.
    """
    st.markdown(
        f""" <style> .stApp {{ background: url("https://img.freepik.com/free-vector/crystal-textured-background-illustration_53876-81310.jpg?w=740&t=st=1689104008~exp=1689104608~hmac=c03ddbe8b38e2bf2d12ce8a1e7943611736829bc868fccbac8532795156843da"); background-size: cover }} </style> """,
        unsafe_allow_html=True
    )


def test_anomaly(dict_models, df_train, df_test, kpi):
    df_test['anomalies'] = 0
    for date in list(df_test['date'].unique()):
        for store in list(df_test['store_name'].unique()):
            model = dict_models[store]
            data = df_train[df_train['store_name'] == store]
            data_for_day = df_test[(df_test['store_name'] == store) & (df_test['date'] == date)]
            if len(data_for_day) > 0:
                data_processed = data.copy()
                data_for_day_processed = data_for_day.copy()

                data_processed = data_processed.drop(columns=['anomalies'])
                data_processed, data_for_day_processed = data_processed.align(data_for_day_processed, axis=1,
                                                                              fill_value=0)
                data_for_day_processed_for_prediction = data_for_day_processed.drop(
                    columns=['Unnamed: 0', 'date', 'store_id','store_name', 'orders_revenue_usd', f'mean_{kpi}_by_industry','industry','less_than_2_weeks_before_holiday',
                             'anomalies'])
                if data_for_day_processed_for_prediction.shape[1] == model[1]:
                    r = model[0].predict(data_for_day_processed_for_prediction)
                    if model[0].__str__()[0:15] == 'IsolationForest':
                        if r[0] == -1:
                            r = 1
                        else:
                            r = 0
                    else:
                        r = r[0]
                    data_for_day['anomalies'] = r
                    df_test.loc[(df_test['store_name'] == store) & (df_test['date'] == date), 'anomalies'] = r
                    if r == 1:
                        all_result[(kpi, store, date)] = pd.concat([data, data_for_day], axis=0, ignore_index=True)

                        # run_anomaly_detection_app(data, data_processed, data_for_day, data_for_day_processed, kpi)
        df_train = pd.concat([df_train, df_test[df_test['date'] == date]])


# def run_anomaly_detection_app(data, data_processed, data_for_day, data_for_day_processed, kpi):


def run_anomaly_detection_app(data_dict):
    set_bg_url()
    kpis = set([k[0] for k in data_dict.keys()])

    kpi = st.sidebar.selectbox(
        'Select KPI',
        kpis)

    stores = set([s[1] for s in data_dict.keys() if s[0] == kpi])

    store_chose = st.sidebar.selectbox(
        'Select Store',
        stores)

    dates = sorted(set([d[2] for d in data_dict.keys() if d[0] == kpi and d[1] == store_chose]))

    date_chose = st.sidebar.selectbox(
        'Select Date',
        dates)
    the_key = (kpi, store_chose, date_chose)
    if kpi == 'Abandoned_orders':
        kpi_name = 'Abandoned orders'
    elif kpi == 'Sessions':
        kpi_name = 'Sessions'
    elif kpi == 'Conversion_Rate_orders':
        kpi_name = 'Conversion rate'
    else:
        kpi_name = 'Average order value'

    if len(data_dict.keys()) > 0:
        full_data = data_dict[the_key]
        data_sorted = full_data.sort_values('date', ascending=False)
        store_name = full_data['store_name'][0]
        data_for_day = full_data.iloc[-1]
        data = full_data.iloc[:-1]

        # Create the Streamlit app
        st.title(':rotating_light: :blue[Anomaly Detection] Report :rotating_light:')
        st.header(f'Store Name: {store_name}')
        st.header(f"Date: {date_chose}")
        st.header(f"KPI: {kpi_name} ")

        # create sidebar with select box
        time_period = ['Week', '2 weeks', 'Month', 'Year']
        st.sidebar.markdown('')
        time_period = st.sidebar.selectbox(
            'Select Time Period:',
            time_period,
            key=store_name + f"{date_chose}"
        )

        if time_period == 'Week':
            samples = 7
        elif time_period == '2 weeks':
            samples = 14
        elif time_period == 'Month':
            samples = 30
        else:
            samples = 365

        # Select the relevant data for the graph
        last_samples = data_sorted.head(samples)  # Assuming the data is sorted by date in descending order

        # Extract the necessary data for plotting
        dates = last_samples['date']
        revenue = last_samples['orders_revenue_usd']
        kpi_last_samples = last_samples[kpi]
        industry_and_shopify_plan = last_samples[f'mean_{kpi}_by_industry']
        industry_name = last_samples['industry']
        # Calculate the correlations
        correlation_revenue_kpi = revenue.corr(kpi_last_samples)
        correlation_kpi_industry_plan = kpi_last_samples.corr(industry_and_shopify_plan)

        if kpi == "Abandoned_orders":
            st.write(
                f"While reviewing your store's data over the last days, we discovered some unusual activity with your Started checkout ⇨ Abandoned checkout conversions that we want to bring to your attention. For a quick explainer, your site's conversion funnel is based on 4 different stages — customers viewing a page, adding products to their cart, starting a checkout, and completing their transaction. The Started checkout ⇨ Made a transaction conversion rate is the ratio of checkouts started compared to the total number of completed purchases. With that out of the way, let's dive into the details. Take a look at your last {samples} samples 'Started checkout ⇨ Abandoned checkout' conversion rate.")

        if kpi == 'Sessions':
            st.write(f"While reviewing your store's data over the last days, we discovered some unusual activity with your Sessions.A Session is happen when a user enter to your website. As we know, more sessions may lead to more sales. For a quick explainer, your site's conversion funnel is based on 4 different stages — customers viewing a page, adding products to their cart, starting a checkout, and completing their transaction. Your problem is in the first stage in your site's conversion funnel.  With that out of the way, let's dive into the details. Take a look at your last {samples} samples")
        if kpi == 'AOV_usd':
            st.write(f"While reviewing your store's data over the last days, we discovered some unusual activity with your AOV. In ecommerce analytics, AOV refers to the average amount of money customers spend when making a purchase on your online store. It's an essential metric that helps gauge the effectiveness of your sales strategies and customer behavior. With that out of the way, let's dive into the details.  Take a look at your last {samples} samples")
        if kpi == 'Conversion_Rate_orders':
            st.write(
                f"While reviewing your store's data over the last days, we discovered some unusual activity with your Conversion Rate.A conversion rate is the percentage of visitors to your website who complete a transaction. As we know, not all the visitors in the website are ending with purchase. For a quick explainer, your site's conversion funnel is based on 4 different stages — customers viewing a page, adding products to their cart, starting a checkout, and completing their transaction. With that out of the way, let's dive into the details. Take a look at your last {samples} samples")

        if kpi_last_samples.iloc[0] < kpi_last_samples[1:].mean():
            st.write(f"Your {kpi_name} from yesterday was significantly lower than your current baseline.")
        if kpi_last_samples.iloc[0] > kpi_last_samples[1:].mean():
            st.write(f"Your {kpi_name} from yesterday was significantly higher than your current baseline.")
        if kpi_name == 'Conversion rate' or 'Abandoned orders':
            st.write(f"Check out yesterday's performance against your average baseline: {round(last_samples[kpi].mean(), 4)}")
        else:
            st.write(f"Check out yesterday's performance against your average baseline: {round(last_samples[kpi].mean(), 0)}")
        st.subheader(f'Last {samples} Samples - {kpi_name}')

        # Plot the line chart for the last samples of KPI
        chart_data = pd.DataFrame({'Date': dates, f'{kpi_name} rate': kpi_last_samples, 'Anomalies': last_samples['anomalies']})
        c = alt.Chart(chart_data).mark_line(point=True).encode(
            x='Date',
            y=f'{kpi_name} rate',
            color=alt.Color('Anomalies:N', scale=alt.Scale(domain=[0, 1], range=['blue', 'red'])),
            tooltip=['Date', f'{kpi_name} rate', 'Anomalies']).configure_axisX(labelAngle=45)

        st.altair_chart(c, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        if kpi_name == 'Conversion rate' or 'Abandoned orders':
            col1.metric("Mean", f'{round(last_samples[kpi].mean(), 4)}')
            col2.metric("Median", f'{round(last_samples[kpi].median(), 4)}')
            col3.metric("Standard Deviation", f'{round(last_samples[kpi].std(), 4)}')
        else:
            col1.metric("Mean", f'{round(last_samples[kpi].mean(), 0)}')
            col2.metric("Median", f'{round(last_samples[kpi].median(), 0)}')
            col3.metric("Standard Deviation", f'{round(last_samples[kpi].std(), 0)}')

        st.write(
            "This red marker represent a day your conversion rate was anomaly value, relate to your baseline for that point in time.")
        if kpi == "Abandoned_orders":
            st.write(
                "Here's the thing — checkouts can be tricky for Shopify merchants. It's where a lot of complications can come up for your customers, and any friction can quickly turn into cart abandonment at the most costly time for your store. The best problem-solving strategy is to track relevant metrics and measure your analytics to figure out where customers are leaving and why. So... why are we telling you all this? From now on, we've got a close eye on your store's performance and will let you know about any concerning cases in real time. That way, you can catch any problem quickly, so you don't lose out on sales!")
        if kpi == 'Conversion_Rate_orders':
            st.write(
                "This anomaly indicates a notable departure from the usual Conversion Rate trend, which could be due to factors such as changes in website design, user experience, or marketing efforts. These issues are better be tracked and fix instantly to prevent loss of sales - More details on how to solve this are at the end of the report.So... why are we telling you all this? From now on, we've got a close eye on your store's performance and will let you know about any concerning cases in real time. That way, you can catch any problem quickly, so you don't lose out on sales!")
        if kpi == 'Sessions':
            st.write("We observed a remarkable deviation from the typical pattern in sessions KPI, suggesting an unusual case in user activity within that timeframe. Identifying and understanding such anomalies is pivotal for optimizing your ecommerce strategies and ensuring seamless user experiences.So... why are we telling you all this? From now on, we've got a close eye on your store's performance and will let you know about any concerning cases in real time. That way, you can catch any problem quickly, so you don't lose out on sales!")
        if kpi == "AOV_usd":
            st.write("This anomaly indicates a noteworthy departure from the usual AOV pattern, suggesting a change in customer spending behavior or a unique event that influenced purchasing decisions. Understanding anomalies like this is crucial for optimizing your ecommerce strategies and maximizing revenue. So... why are we telling you all this? From now on, we've got a close eye on your store's performance and will let you know about any concerning cases in real time. That way, you can catch any problem quickly, so you don't lose out on sales!")

        if round(correlation_kpi_industry_plan, 2) != 1 and revenue.mean() != 0:
            col1, col2 = st.columns(2)
            with col1:
                col1.metric("Correlation Revenue-KPI: ", f'{round(correlation_revenue_kpi, 2)}')
                # Plot the line chart for revenue
                st.subheader(f'Last {samples} days - revenue in USD')
                revenue_data = pd.DataFrame({'Date': dates, 'Revenue': revenue})

                revenue_chart = alt.Chart(revenue_data).mark_line().encode(
                    x="Date",
                    y='Revenue'
                ).configure_axisX(labelAngle=45)
                st.altair_chart(
                    revenue_chart.interactive(),
                    use_container_width=True
                )
                if correlation_revenue_kpi > 0:
                    st.write(f"As you can see the correlation in {kpi_name} KPI with your store's revenue is positive.")
                    if kpi == "Abandoned_orders":
                        st.write(f"Although your revenue is correlated with {kpi_name} KPI, our system recognized anomaly behavior. It may happen because other factors in your store.")
                    elif kpi == "Sessions":
                        st.write(
                            f"As we know, the {kpi_name} KPI is influence the sales. Sales impact revenue, so when there is a positive correlation, it strengthens our assertion that the day is an anomaly.")
                    else:
                        st.write(
                            f"As we know, the {kpi_name} KPI is influenced by sales. Sales impact revenue, so when there is a positive correlation, it strengthens our assertion that the day is an anomaly.")
                else:
                    st.write(f"As you can see the correlation in {kpi_name} KPI with your store's revenue is negative.")
                    if kpi == "Abandoned_orders":
                        st.write(
                            f"As we know, the {kpi_name} KPI is influenced by sales. Sales impact revenue, so when there is a negative correlation, it strengthens our assertion that the day is an anomaly.")
                    else:
                        st.write(f"Although your revenue is not correlated with {kpi_name} KPI, our system recognized anomaly behavior. It may happen because other factors in your store.")

            with col2:
                col2.metric("Correlation KPI store-KPI industry: ", f'{round(correlation_kpi_industry_plan, 2)}')
                # Plot the line chart for mean KPI by industry
                st.subheader(f'Last {samples} days mean {kpi_name} by industry')
                chart_data = pd.DataFrame({'Date': dates, f'mean {kpi_name} by industry': industry_and_shopify_plan})
                data_chart = alt.Chart(chart_data).mark_line().encode(
                    x="Date",
                    y=f'mean {kpi_name} by industry'
                ).configure_axisX(labelAngle=45)
                st.altair_chart(
                    data_chart.interactive(),
                    use_container_width=True
                )
                if correlation_kpi_industry_plan > 0:
                    st.write(
                        f"As you can observe, there is a positive correlation between the {kpi_name} KPI and the mean value of the same KPI within the store's industry")
                    st.write(f"This might indicate that the anomaly value is a result of a trend across the entire {industry_name} industry.")
                else:
                    st.write(
                        f"As you can observe, there is a negative correlation between the {kpi_name} KPI and the mean value of the same KPI within the store's industry")
                    st.write(f"This might indicate that the anomaly value is a result of a unique reasons in your store, and not in the entire {industry_name} industry.")


        elif revenue.mean() != 0:
            # Plot the line chart for revenue
            st.subheader(f'Last {samples} days - revenue in USD')
            st.subheader(f"Correlation Revenue-KPI: {round(correlation_revenue_kpi, 2)}")
            revenue_data = pd.DataFrame({'Date': dates, 'Revenue': revenue})

            revenue_chart = alt.Chart(revenue_data).mark_line().encode(
                x="Date",
                y='Revenue'
            ).configure_axisX(labelAngle=45)
            st.altair_chart(
                revenue_chart.interactive(),
                use_container_width=True
            )
            if correlation_revenue_kpi > 0:
                st.write(f"As you can see the correlation in {kpi_name} KPI with your store's revenue is positive. ")
                if kpi == "Abandoned_orders":
                    st.write(f"Although your revenue is correlated with {kpi_name} KPI, our system recognized anomaly behavior. It may happen because other factors in your store.")
                elif kpi == "Sessions":
                    st.write(
                        f"As we know, the {kpi_name} KPI is influence the sales. Sales impact revenue, so when there is a positive correlation, it strengthens our assertion that the day is an anomaly.")
                else:
                    st.write(
                        f"As we know, the {kpi_name} KPI is influenced by sales. Sales impact revenue, so when there is a positive correlation, it strengthens our assertion that the day is an anomaly.")
            else:
                st.write(f"As you can see the correlation in {kpi_name} KPI with your store's revenue is negative. ")
                if kpi == "Abandoned_orders":
                    st.write(
                        f"As we know, the {kpi_name} KPI is influenced by sales. Sales impact revenue, so when there is a negative correlation, it strengthens our assertion that the day is an anomaly.")
                else:
                    st.write(f"Although your revenue is not correlated with {kpi_name} KPI, our system recognized anomaly behavior. It may happen because other factors in your store.")
        elif correlation_kpi_industry_plan != 1:
            st.subheader(f'Last {samples} days mean {kpi_name} by industry')
            st.subheader("Correlation KPI store-KPI industry: ", f'{round(correlation_kpi_industry_plan, 2)}')
            # Plot the line chart for mean KPI by industry
            chart_data = pd.DataFrame({'Date': dates, f'mean {kpi_name} by_industry': industry_and_shopify_plan})
            data_chart = alt.Chart(chart_data).mark_line().encode(
                x="Date",
                y=f'mean {kpi_name} by_industry'
            ).configure_axisX(labelAngle=45)
            st.altair_chart(
                data_chart.interactive(),
                use_container_width=True
            )
            if correlation_kpi_industry_plan > 0:
                st.write(
                    f"As you can observe, there is a positive correlation between the {kpi_name} KPI and the mean value of the same KPI within the store's industry")
                st.write(f"This might indicate that the anomaly value is a result of a trend across the entire {industry_name} industry.")
            else:
                st.write(
                    f"As you can observe, there is a negative correlation between the {kpi_name} KPI and the mean value of the same KPI within the store's industry")
                st.write(f"This might indicate that the anomaly value is a result of a unique reasons in your store, and not in the entire {industry_name} industry.")

        full_data_pie = full_data.copy()
        data_sorted_pie = full_data_pie.sort_values('date', ascending=False)
        data_sorted_pie = data_sorted_pie.head(samples)

        # Pie chart for the distribution of less_than_2_weeks_before_holiday
        holiday_distribution = data_sorted_pie.groupby(['less_than_2_weeks_before_holiday']).sum()['orders_revenue_usd']
        labels = holiday_distribution.index.tolist()
        current_holiday = data_for_day['less_than_2_weeks_before_holiday']
        if holiday_distribution.sum() != 0:
            percentages = holiday_distribution.values.tolist() / holiday_distribution.sum()
            explode = np.zeros(len(percentages))
            explode[labels.index(current_holiday)] = 0.1

            df = pd.DataFrame({'holiday': labels, 'percentages': percentages})
            col1, col2, pie_col, col4, col5 = st.columns([1, 1, 2, 1, 1])
            with pie_col:
                st.subheader(f'Last {samples} days - revenue percentages by holidays')
                fig1, ax1 = plt.subplots()
                fig1 = px.pie(df, values='percentages', names='holiday')
                fig1.update_traces(pull=explode)
                ax1.set_title(f'Last {samples} days distribution of revenue per holiday')
                ax1.axis('equal')
                pie_col.write(fig1)

        if current_holiday != "Regular Day":
            date_index = pd.DatetimeIndex([data_for_day['date']])
            data_for_day_year = date_index.year[0]
            data_year = data_for_day_year - 1
            same_period_data = data[
                (data['less_than_2_weeks_before_holiday'] == data_for_day['less_than_2_weeks_before_holiday'])]

            same_period_data = same_period_data[pd.DatetimeIndex(same_period_data['date']).year == data_year]
            kpi_last_year = same_period_data[kpi]
            if len(kpi_last_year) > 0:
                # Plot the box plot for KPI last year
                st.subheader(f'Same holiday last year - {kpi_name}')
                fig = px.box(kpi_last_year)
                # Display the plot in the Streamlit app
                st.plotly_chart(fig, theme=None, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                if kpi_name == 'Conversion rate' or 'Abandoned orders':
                    col1.metric("Mean", f'{round(kpi_last_year.mean(), 4)}')
                    col2.metric("Median", f'{round(kpi_last_year.median(), 4)}')
                    col3.metric("Standard Deviation", f'{round(kpi_last_year.std(), 4)}')
                else:
                    col1.metric("Mean", f'{round(kpi_last_year.mean(), 0)}')
                    col2.metric("Median", f'{round(kpi_last_year.median(), 0)}')
                    col3.metric("Standard Deviation", f'{round(kpi_last_year.std(), 0)}')


                st.write(
                    "Given that the anomaly detection occurred close to holiday days, we can examine the events and trends that occurred in the period around the same holiday last year for the same KPI.")

        # show dataframe is checkbox selected
        st.dataframe(data=last_samples.iloc[:, 2:], use_container_width=True)

        st.subheader("How to act")
        if kpi == 'Abandoned_orders':
            if kpi_last_samples.iloc[0] > kpi_last_samples[1:].mean():
                st.write(
                    "The checkout process has 3 main steps, each with the potential for a customer to abandon the process entirely.")
                st.write(
                    "**Step 1 – Personal info** - Make sure you're asking **only necessary details** to keep checkout as easy and simple as possible. Requiring customers to create an account or give excessive personal information can increase abandonment.")
                st.write(
                    "**Step 2 – Shipping method** - Make sure to give your customers the information they need to calculate shipping costs, fees, taxes, and delivery time **before** checkout. That way customers won't jump ship when they realize their purchase is going to be more expensive than they originally thought.")
                st.write(
                    "**Step 3 – Payment** - Check your analytics to see if there's a decrease in the use of a particular payment method. This could indicate some type of technical problem that needs fixing. Also, make sure there's no issue with the country code from which the failed transactions were made.")
                st.write(
                    "In addition to these potential hiccups, since the checkout process is the “last mile of code” on your store, general issues you didn't even know about can drain into this phase and cause unexpected errors. This could be due to inventory issues or payment methods that were set up incorrectly.")
                st.write(
                    "Friction is also created when users add coupons and discounts, select a different currency, or click agreement boxes. Just remember — the more you ask of your customers, the more likely they are to abandon the checkout.")
                st.write(
                    "And wait… there's more. If your payment provider isn't on top of things, you may experience a higher-than-average number of payment failures, retries, and false positive fraud alerts. Any of these issues can prevent a transaction from being approved, which may prompt the customer to abandon the transaction entirely.")
            else:
                st.write("Acknowledge the positive decline in Abandoned Orders KPI – your efforts to improve user experience are yielding results. To maintain this progress, continue refining checkout processes, analyzing abandonment patterns to identify pain points, and optimizing marketing strategies based on the decreased abandoned orders. Your commitment to enhancing user journeys is evident, and we're optimistic about the ongoing positive impact on the Abandoned Orders KPI.")
        if kpi == 'Sessions':
            if kpi_last_samples.iloc[0] < kpi_last_samples[1:].mean():
                st.write(
                    "  * If your website's content and design have become outdated or stale, visitors may lose interest and not find a reason to return.- Update your website's content, images, and design to make it more visually appealing and relevant to your target audience. Regularly refresh your content with new and engaging information to keep visitors coming back. Conduct usability tests to ensure user-friendly navigation.")
                st.write(
                    "  * The investment in advertising and marketing is not good enough, so people are not exposed enough to your website - Try to look for other internet marketing sources or increase your investment in marketing")
            else:
                st.write(
                    "Celebrate the remarkable increase in Sessions KPI – your dedication is shining brightly! To maintain and build on this success, ensure users remain engaged with compelling content, delve into session data for valuable insights, and refine marketing strategies based on the patterns observed in today's sessions. Your hard work is clearly making a significant impact, and we're eager to witness the continued growth of your Sessions KPI.")

        if kpi == 'AOV_usd':
            if kpi_last_samples.iloc[0] < kpi_last_samples[1:].mean():
                st.write(
                    "AOV decrease can often be attributed to changes that impact user spending behavior or overall purchasing trends. Several factors can lead to a drop in AOV:")
                st.write(
                    "  * Offering heavy discounts or promotions: If you've been offering significant discounts, consider analyzing their impact on AOV. You might want to strike a balance between attracting customers and maintaining a healthy AOV.")
                st.write(
                    "  * Introducing low-cost products that dilute the average value: While expanding your product range can be great, ensure that lower-priced items don't disproportionately lower the overall AOV. Consider cross-selling or bundling strategies to encourage higher-value purchases.")
                st.write(
                    "  * Changes in the product mix or assortment: When introducing new products or discontinuing certain items, monitor their effect on AOV. Ensure that new additions align with your target AOV and customer preferences.")
                st.write(
                    "  * Seasonal or external factors: Sometimes, factors like holiday seasons, economic conditions, or global events can impact consumer behavior. While some changes might be temporary, adapt your marketing and pricing strategies to mitigate AOV fluctuations.")

            else:
                st.write(
                    "Embrace the impressive rise in Average Order Value (AOV) – your strategies are yielding fantastic outcomes! To sustain this progress, continue to offer value-driven products, analyze purchasing patterns for valuable insights, and refine marketing strategies based on the successful AOV observed today. Your diligent efforts are clearly making a significant impact on the bottom line, and we're excited to see the AOV increase even further.")

        if kpi == 'Conversion_Rate_orders':
            if kpi_last_samples.iloc[0] < kpi_last_samples[1:].mean():
                st.write(
                    "Conversion Rate sharp changed is usually a result of website modifications that in a way impairs its online performance or user experience. A number of factors can cause this to occur:")
                st.write("Installing a new application, updating the theme , changing information or pictures on product detail pages.")
                st.write(
                    "If you've made any recent updates like this to your site that match up to these drops in your conversion rates, consider making adjustments to get your store back on track.")
            else:
                st.write(
                    "Celebrate the remarkable increase in Conversion Rate – your hard work is paying off! To maintain this momentum, focus on optimizing user experiences, analyze conversion funnel insights, and refine marketing strategies based on the successful conversion rate achieved today. Your dedication is clearly driving outstanding results, and we're excited to see the conversion rate continue to rise.")


def delete_csv_file(file_name):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the file path by joining the current directory and file name
    file_path = os.path.join(current_dir, file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)


def main():
    st.set_page_config(
        page_title="Real-Time Anomaly Detection Dashboard",
        layout="wide",
    )
    global all_result
    all_result = {}
    for kpi in ['Sessions', 'AOV_usd', 'Conversion_Rate_orders', 'Abandoned_orders']:
        kpi_x, kpi_test, kpi_name, kpi_dict_models = pd.read_csv(f'{kpi}_X.csv'), pd.read_csv(
            f'{kpi}_test.csv'), kpi, f'my_dict_{kpi}.pkl'

        # read the dictionary from the file in another file
        with open(kpi_dict_models, 'rb') as f:
            kpi_dict_models = pickle.load(f)
        test_anomaly(kpi_dict_models, kpi_x, kpi_test, kpi_name)
    run_anomaly_detection_app(all_result)


if __name__ == '__main__':
    main()
