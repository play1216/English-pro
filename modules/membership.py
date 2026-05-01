import pandas as pd
import plotly.express as px
import streamlit as st


def render_membership_panel(subscription, afdian_client, panel_key="default"):
    st.subheader("💳 会员与额度")

    st.metric("价格", f"¥{subscription.get_price()}/月")
    st.metric("月上限", f"{subscription.get_limit()}篇")
    st.metric("免费额度", f"{subscription.free_remaining()}/{subscription.get_free_limit()}")

    if subscription.is_subscribed():
        used = subscription.current_usage()
        remaining = subscription.remaining()
        df_usage = pd.DataFrame({"类型": ["本月已用", "本月剩余"], "数量": [used, remaining]})
        st.success("已开通爱发电会员")
        membership_status = subscription.get_membership_status()
        if membership_status.get("expires_at"):
            st.caption(f"会员有效期至：{membership_status['expires_at']}")
        if membership_status.get("is_expiring_soon"):
            st.warning(f"会员将在 {membership_status.get('days_left', 0)} 天内到期，建议提前续费。")
        payment = subscription.get_last_payment()
        if payment and payment.get("order_no"):
            st.caption(
                f"最近绑定订单：{payment['order_no']} | 金额：{payment.get('amount', '未知')} | 套餐：{payment.get('plan_name', '未知')}"
            )
    else:
        used = subscription.free_used()
        remaining = subscription.free_remaining()
        df_usage = pd.DataFrame({"类型": ["已用免费", "剩余免费"], "数量": [used, remaining]})
        if subscription.has_free_quota():
            st.info(f"当前还有 {remaining} 次免费批改机会。")
        else:
            st.warning("3 次免费额度已用完，开通爱发电会员后可继续使用。")

    fig_usage = px.bar(
        df_usage,
        x="数量",
        y="类型",
        orientation="h",
        text="数量",
        color="类型",
        height=220,
    )
    fig_usage.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    )
    fig_usage.update_layout(
        showlegend=False,
        margin=dict(l=10, r=36, t=10, b=10),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    st.plotly_chart(
        fig_usage,
        use_container_width=True,
        config={"displayModeBar": False, "staticPlot": True, "scrollZoom": False},
    )

    if subscription.is_subscribed():
        return

    st.markdown("**爱发电开通会员**")
    if not afdian_client.is_configured():
        st.error("爱发电配置不完整，请先配置 AFDIAN_USER_ID 与 AFDIAN_TOKEN。")
        return
    if afdian_client.payment_url:
        st.markdown(f"[前往爱发电支付]({afdian_client.payment_url})")
    else:
        st.caption("请在 `.streamlit/secrets.toml` 中配置 `AFDIAN_PAYMENT_URL`。")
    st.caption("支付完成后，复制爱发电订单号并在下方提交验证。")

    with st.form(f"afdian_verify_form_{panel_key}"):
        order_no = st.text_input("支付后填写爱发电订单号")
        verify_submitted = st.form_submit_button("验证订单并开通会员", use_container_width=True)

    if verify_submitted:
        success, message, payment_info = afdian_client.verify_order(order_no)
        if not success:
            st.error(message)
        else:
            activated, activate_message = subscription.activate_membership(order_no, payment_info)
            if activated:
                st.success(activate_message)
                st.rerun()
            else:
                st.error(activate_message)
