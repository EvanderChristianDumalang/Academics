using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AchievementController : MonoBehaviour
{
    private static AchievementController _instance = null;
    public static AchievementController Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<AchievementController>();
            }
            return _instance;
        }
    }

    private List<int> _goldMilestones = new List<int>();
    public List<int> GoldMilestones
    {
        get
        {
            return _goldMilestones;
        }
    }

    [SerializeField] private Transform _popUpTransform;
    [SerializeField] private Text _popUpText;
    [SerializeField] private float _popUpShowDuration = 3f;
    [SerializeField] private List<AchievementData> _achievementList;

    private float _popUpShowDurationCounter;

    private void Awake()
    {
        foreach(AchievementData achievement in _achievementList)
        {
            if(achievement.Type == AchievementType.GetGoldAmount)
            {
                _goldMilestones.Add(int.Parse(achievement.Value));
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (_popUpShowDurationCounter > 0)
        {
            _popUpShowDurationCounter -= Time.unscaledDeltaTime;
            _popUpTransform.localScale = Vector3.LerpUnclamped(_popUpTransform.localScale, Vector3.one, 0.5f);
        }
        else
        {
            _popUpTransform.localScale = Vector2.LerpUnclamped(_popUpTransform.localScale, Vector3.right, 0.5f);
        }
    }

    public void UnlockAchievement(AchievementType type, string value)
    {
        AchievementData achievement = _achievementList.Find(a => a.Type == type && a.Value == value);
        if (achievement != null && !achievement.IsUnlocked)
        {
            achievement.IsUnlocked = true;

            ShowAchivementPopUp(achievement);
        }
    }

    public void RemoveGoldMilestones(List<int> milestones)
    {
        foreach(int value in milestones)
        {
            if (_goldMilestones.Contains(value))
            {
                _goldMilestones.Remove(value);
            }
        }
    }

    private void ShowAchivementPopUp(AchievementData achievement)
    {
        _popUpText.text = achievement.Title;

        _popUpShowDurationCounter = _popUpShowDuration;

        _popUpTransform.localScale = Vector2.right;
    }
}

[System.Serializable]

public class AchievementData
{
    public string Title;
    public AchievementType Type;
    public string Value;
    public bool IsUnlocked;
}



public enum AchievementType
{
    UnlockResource,
    GetGoldAmount
}